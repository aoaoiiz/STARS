from __future__ import annotations

import argparse
import json
import math
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
from creative_video_exp.config import (
    FORMAL_RISK_TERMS,
    validate_videollama2_runtime_identity,
)

FORMAL_REWARD_WEIGHTS = {
    "alignment_weight": 6.0 / 17.0,
    "readability_weight": 3.0 / 17.0,
    "rhythm_weight": 3.0 / 17.0,
    "control_weight": 4.0 / 17.0,
    "risk_weight": 1.0 / 17.0,
}

MATRIX_SUMMARY_SCHEMA = "stars_matrix_summary_v1"
PAPER_METRICS = (
    "reward",
    "mv_align",
    "csr",
    "semantic_point_coverage",
    "cider_lite",
    "rep",
)
METHOD_KEYS = ("direct_generation_c1", "stars_best_of_4")
UPPER_BOUND_DEFINITION = {
    "name": "Metric-wise Best@4 Upper Bound",
    "pool": "valid candidates among C1-C4",
    "selection": "independent best candidate for each metric",
    "single_realizable_candidate": False,
    "deployable": False,
}
ACCOUNTING_KEYS = (
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
    "real_video_rate",
    "multimodal_generation_rate",
)
EFFICIENCY_KEYS = (
    "mean_sampling_seconds",
    "mean_generation_seconds",
    "mean_generation_requests",
    "mean_generation_input_tokens",
    "mean_generation_output_tokens",
    "mean_generation_total_tokens",
    "mean_direct_generation_seconds",
    "mean_stars_seconds",
    "mean_direct_generation_input_tokens",
    "mean_stars_input_tokens",
    "mean_direct_generation_output_tokens",
    "mean_stars_output_tokens",
    "mean_direct_generation_total_tokens",
    "mean_stars_total_tokens",
    "mean_direct_generation_requests",
    "mean_stars_generation_requests",
    "mean_reward_encoding_seconds",
    "mean_reward_scoring_seconds",
    "mean_metric_evaluation_seconds",
    "mean_total_pipeline_seconds",
    "mean_vlm_server_peak_allocated_mib",
    "mean_vlm_server_peak_reserved_mib",
    "mean_reward_runner_peak_allocated_mib",
    "mean_reward_runner_peak_reserved_mib",
)

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
    _validate_project_relative_option(args.config_dir, "--config-dir")
    _validate_project_relative_option(args.output_root, "--output-root")
    config_dir = PROJECT_ROOT / args.config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = PROJECT_ROOT / args.output_root
    summary_dir.mkdir(parents=True, exist_ok=True)
    existing_rows = _load_existing_summary_rows(summary_dir / "summary.json")
    current_rows = []
    model_bindings, reward_binding = _prepare_checkpoint_identities(args.models)

    for dataset_key in args.datasets:
        for model_key in args.models:
            config = build_config(
                dataset_key,
                model_key,
                args,
                model_identity=model_bindings[model_key]["identity"],
                model_runtime_identity=model_bindings[model_key][
                    "runtime_identity"
                ],
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
            current_rows.append(
                _summary_row(
                    dataset_key,
                    model_key,
                    config_path,
                    metrics_path,
                    metrics,
                )
            )

    rows = _merge_summary_rows(existing_rows, current_rows)
    (summary_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
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
    model_runtime_identity: dict[str, Any] | None = None,
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
            "risk_terms": list(FORMAL_RISK_TERMS),
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
                    "temperature": 0.3,
                    "request_timeout_sec": 300,
                    "retry_count": 2,
                    "seed": 42,
                    "checkpoint_identity": model_identity,
                    "runtime_identity": model_runtime_identity or {},
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
        service_identity = _query_local_service_identity(
            endpoint_url,
            api_key_env=model.get("api_key_env", "VLM_API_KEY"),
            expected_model_id=model["name"],
        )
        observed_identity = service_identity["checkpoint_identity"]
        if observed_identity != expected_identity:
            raise RuntimeError(
                f"Live service checkpoint identity differs from {manifest_env}."
            )
        runtime_identity = service_identity["runtime_identity"]
        if model_key == "videollama2_7b_16f":
            validate_videollama2_runtime_identity(runtime_identity)
        bindings[model_key] = {
            "identity": expected_identity,
            "runtime_identity": runtime_identity,
            "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        }
    return bindings, {
        "identity": reward_identity,
        "manifest_path": str(Path(reward_manifest_path).expanduser().resolve()),
    }


def _query_local_service_identity(
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
    max_frames = matches[0].get("max_frames")
    if isinstance(max_frames, bool) or max_frames != 16:
        raise RuntimeError("Service must advertise max_frames=16.")
    frame_input_policy = matches[0].get("frame_input_policy")
    if frame_input_policy != "one_to_sixteen_data_images_without_service_resampling":
        raise RuntimeError(
            "Service must advertise the no-service-resampling frame policy."
        )
    runtime_identity = matches[0].get("runtime_identity", {})
    if not isinstance(runtime_identity, dict):
        raise RuntimeError("Service runtime identity must be an object.")
    return {
        "checkpoint_identity": identity,
        "runtime_identity": runtime_identity,
    }


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


def _summary_row(
    dataset: str,
    model: str,
    config_path: Path,
    metrics_path: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    accounting = {
        key: _required_value(metrics, key, "metrics")
        for key in ACCOUNTING_KEYS
    }
    if accounting["evidence_eligible"] is not True:
        raise RuntimeError("Metrics are not eligible for the formal matrix summary.")
    success_rates = _required_mapping(metrics, "method_output_success_rates", "metrics")
    conditional = _required_mapping(metrics, "method_conditional_means", "metrics")
    effective = _required_mapping(metrics, "method_effective_means", "metrics")
    methods: dict[str, Any] = {}
    for method in METHOD_KEYS:
        methods[method] = {
            "selection_success_rate": _required_number(
                success_rates,
                method,
                "method_output_success_rates",
            ),
            "conditional_means": _metric_values(
                _required_mapping(conditional, method, "method_conditional_means"),
                f"method_conditional_means.{method}",
                allow_none=True,
            ),
            "effective_means": _metric_values(
                _required_mapping(effective, method, "method_effective_means"),
                f"method_effective_means.{method}",
                allow_none=False,
            ),
        }
    upper_source = _required_mapping(
        metrics,
        "metric_wise_best_of_4_upper_bound",
        "metrics",
    )
    upper_definition = _required_mapping(
        metrics,
        "metric_wise_best_of_4_upper_bound_definition",
        "metrics",
    )
    _validate_upper_bound_definition(upper_definition)
    upper = {
        "selection_success_rate": _required_number(
            upper_source,
            "selection_success_rate",
            "metric_wise_best_of_4_upper_bound",
        ),
        "conditional_means": _metric_values(
            _required_mapping(
                upper_source,
                "conditional_means",
                "metric_wise_best_of_4_upper_bound",
            ),
            "metric_wise_best_of_4_upper_bound.conditional_means",
            allow_none=True,
        ),
        "effective_means": _metric_values(
            _required_mapping(
                upper_source,
                "effective_means",
                "metric_wise_best_of_4_upper_bound",
            ),
            "metric_wise_best_of_4_upper_bound.effective_means",
            allow_none=False,
        ),
    }
    efficiency = {
        key: _required_number(metrics, key, "metrics")
        for key in EFFICIENCY_KEYS
    }
    protocol = _required_mapping(metrics, "protocol", "metrics")
    fingerprint = _required_value(protocol, "protocol_fingerprint", "metrics.protocol")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RuntimeError("metrics.protocol.protocol_fingerprint must be a non-empty string.")
    row = {
        "schema_version": MATRIX_SUMMARY_SCHEMA,
        "dataset": dataset,
        "model": model,
        "config_path": _portable_path(config_path),
        "metrics_path": _portable_path(metrics_path),
        "protocol_fingerprint": fingerprint,
        "accounting": accounting,
        "methods": methods,
        "metric_wise_best_of_4_upper_bound_definition": dict(
            upper_definition
        ),
        "metric_wise_best_of_4_upper_bound": upper,
        "efficiency": efficiency,
    }
    _validate_summary_row(row)
    return row


def _load_existing_summary_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Existing matrix summary is invalid JSON: {path}") from exc
    if not isinstance(payload, list):
        raise RuntimeError("Existing matrix summary must be a JSON array.")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            raise RuntimeError("Every existing matrix summary row must be an object.")
        _validate_summary_row(raw)
        key = (str(raw["dataset"]), str(raw["model"]))
        if key in seen:
            raise RuntimeError(f"Duplicate existing matrix summary row: {key}.")
        seen.add(key)
        rows.append(raw)
    return rows


def _merge_summary_rows(
    existing: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {
        (str(row["dataset"]), str(row["model"])): row
        for row in existing
    }
    for row in current:
        _validate_summary_row(row)
        merged[(str(row["dataset"]), str(row["model"]))] = row
    dataset_order = {name: index for index, name in enumerate(DATASETS)}
    model_order = {name: index for index, name in enumerate(MODELS)}
    return sorted(
        merged.values(),
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            model_order[str(row["model"])],
        ),
    )


def _validate_summary_row(row: dict[str, Any]) -> None:
    if row.get("schema_version") != MATRIX_SUMMARY_SCHEMA:
        raise RuntimeError(
            "Existing matrix summary uses an incompatible schema; use a new output root."
        )
    dataset = str(_required_value(row, "dataset", "summary row"))
    model = str(_required_value(row, "model", "summary row"))
    if dataset not in DATASETS or model not in MODELS:
        raise RuntimeError(f"Unknown dataset/model in matrix summary: {(dataset, model)}.")
    fingerprint = _required_value(row, "protocol_fingerprint", "summary row")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RuntimeError("Matrix summary protocol fingerprint is missing.")
    for key in ("config_path", "metrics_path"):
        value = _required_value(row, key, "summary row")
        path = Path(str(value))
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Matrix summary {key} must be project-relative.")
    accounting = _required_mapping(row, "accounting", "summary row")
    _validate_accounting(accounting)
    methods = _required_mapping(row, "methods", "summary row")
    for method in METHOD_KEYS:
        payload = _required_mapping(methods, method, "summary row methods")
        _validate_method_summary(payload, f"summary row {method}")
    upper = _required_mapping(
        row,
        "metric_wise_best_of_4_upper_bound",
        "summary row",
    )
    upper_definition = _required_mapping(
        row,
        "metric_wise_best_of_4_upper_bound_definition",
        "summary row",
    )
    _validate_upper_bound_definition(upper_definition)
    _validate_method_summary(upper, "summary row upper bound")
    direct = methods["direct_generation_c1"]
    stars = methods["stars_best_of_4"]
    direct_success = float(direct["selection_success_rate"])
    stars_success = float(stars["selection_success_rate"])
    upper_success = float(upper["selection_success_rate"])
    if direct_success > stars_success + 1e-6:
        raise RuntimeError("Direct Generation success cannot exceed STARS success.")
    if not _close(stars_success, upper_success):
        raise RuntimeError("STARS and the metric-wise upper bound must share a success rate.")
    if float(accounting["full_candidate_pool_sample_rate"]) > stars_success + 1e-6:
        raise RuntimeError("Full-pool completion cannot exceed STARS success.")
    for means_key in ("conditional_means", "effective_means"):
        stars_reward = stars[means_key]["reward"]
        upper_reward = upper[means_key]["reward"]
        if stars_reward is None or upper_reward is None:
            if stars_reward is not upper_reward:
                raise RuntimeError(
                    "STARS and upper-bound Reward availability is inconsistent."
                )
        elif not _close(float(stars_reward), float(upper_reward)):
            raise RuntimeError(
                "Metric-wise Best@4 Reward must equal STARS Reward."
            )
    efficiency = _required_mapping(row, "efficiency", "summary row")
    for key in EFFICIENCY_KEYS:
        _required_nonnegative_number(
            efficiency,
            key,
            "summary row efficiency",
        )


def _validate_accounting(accounting: dict[str, Any]) -> None:
    requested = _required_int(
        accounting,
        "requested_num_samples",
        "summary row accounting",
        minimum=1,
    )
    completed = _required_int(
        accounting,
        "completed_num_samples",
        "summary row accounting",
    )
    requested_slots = _required_int(
        accounting,
        "requested_candidate_slot_count",
        "summary row accounting",
    )
    valid_slots = _required_int(
        accounting,
        "valid_candidate_slot_count",
        "summary row accounting",
    )
    failed_slots = _required_int(
        accounting,
        "failed_candidate_slot_count",
        "summary row accounting",
    )
    slot_rate = _required_rate(
        accounting,
        "candidate_slot_success_rate",
        "summary row accounting",
    )
    full_pools = _required_int(
        accounting,
        "full_candidate_pool_sample_count",
        "summary row accounting",
    )
    full_pool_rate = _required_rate(
        accounting,
        "full_candidate_pool_sample_rate",
        "summary row accounting",
    )
    unresolved = _required_int(
        accounting,
        "unresolved_generation_failure_count",
        "summary row accounting",
    )
    num_samples = _required_int(
        accounting,
        "num_samples",
        "summary row accounting",
    )
    num_candidates = _required_int(
        accounting,
        "num_candidates",
        "summary row accounting",
    )
    real_video_rate = _required_rate(
        accounting,
        "real_video_rate",
        "summary row accounting",
    )
    multimodal_rate = _required_rate(
        accounting,
        "multimodal_generation_rate",
        "summary row accounting",
    )
    if accounting.get("evidence_eligible") is not True:
        raise RuntimeError("Matrix summary row is not formally eligible.")
    if completed != requested or num_samples != requested:
        raise RuntimeError("Completed and summarized samples must equal requested samples.")
    if requested_slots != requested * 4:
        raise RuntimeError("Requested candidate slots must equal four per sample.")
    if valid_slots + failed_slots != requested_slots:
        raise RuntimeError("Valid and failed slot counts do not match requested slots.")
    if num_candidates != valid_slots:
        raise RuntimeError("Candidate count must equal valid slot count.")
    if full_pools > requested:
        raise RuntimeError("Full-pool count exceeds requested samples.")
    if unresolved != 0:
        raise RuntimeError("An eligible matrix row cannot contain unresolved failures.")
    if not _close(slot_rate, valid_slots / requested_slots):
        raise RuntimeError("Candidate-slot success rate does not match slot counts.")
    if not _close(full_pool_rate, full_pools / requested):
        raise RuntimeError("Full-pool rate does not match its count.")
    if not _close(real_video_rate, 1.0) or not _close(multimodal_rate, 1.0):
        raise RuntimeError("Formal matrix rows require real video and multimodal generation.")


def _validate_method_summary(payload: dict[str, Any], context: str) -> None:
    success = _required_rate(payload, "selection_success_rate", context)
    conditional = _metric_values(
        _required_mapping(payload, "conditional_means", context),
        f"{context}.conditional_means",
        allow_none=True,
    )
    effective = _metric_values(
        _required_mapping(payload, "effective_means", context),
        f"{context}.effective_means",
        allow_none=False,
    )
    if success == 0.0:
        if any(value is not None for value in conditional.values()):
            raise RuntimeError(f"{context} has conditional means without successes.")
    elif any(value is None for value in conditional.values()):
        raise RuntimeError(f"{context} lacks conditional means for successful outputs.")
    for metric in PAPER_METRICS:
        conditional_value = conditional[metric]
        expected = (
            1.0
            if conditional_value is None and metric == "rep"
            else 0.0
            if conditional_value is None
            else float(conditional_value) * success
            + (1.0 - success if metric == "rep" else 0.0)
        )
        if not _close(float(effective[metric]), expected, tolerance=3e-6):
            raise RuntimeError(
                f"{context}.{metric} effective mean does not match its failure penalty."
            )


def _required_mapping(
    payload: dict[str, Any],
    key: str,
    context: str,
) -> dict[str, Any]:
    value = _required_value(payload, key, context)
    if not isinstance(value, dict):
        raise RuntimeError(f"{context}.{key} must be an object.")
    return value


def _required_value(payload: dict[str, Any], key: str, context: str) -> Any:
    if key not in payload:
        raise RuntimeError(f"Missing required field {context}.{key}.")
    return payload[key]


def _required_number(payload: dict[str, Any], key: str, context: str) -> float:
    value = _required_value(payload, key, context)
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


def _required_rate(payload: dict[str, Any], key: str, context: str) -> float:
    number = _required_number(payload, key, context)
    if not 0.0 <= number <= 1.0:
        raise RuntimeError(f"{context}.{key} must be within [0, 1].")
    return number


def _required_int(
    payload: dict[str, Any],
    key: str,
    context: str,
    *,
    minimum: int = 0,
) -> int:
    value = _required_value(payload, key, context)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(
            f"{context}.{key} must be an integer greater than or equal to {minimum}."
        )
    return value


def _close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _metric_values(
    payload: dict[str, Any],
    context: str,
    *,
    allow_none: bool,
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for metric in PAPER_METRICS:
        value = _required_value(payload, metric, context)
        if value is None and allow_none:
            output[metric] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"{context}.{metric} must be numeric.")
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError(f"{context}.{metric} must be finite.")
        if not 0.0 <= number <= 1.0:
            raise RuntimeError(f"{context}.{metric} must be within [0, 1].")
        output[metric] = number
    return output


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"Matrix artifact path is outside the project tree: {resolved}"
        ) from exc


def _validate_upper_bound_definition(payload: dict[str, Any]) -> None:
    if payload != UPPER_BOUND_DEFINITION:
        raise RuntimeError(
            "Metric-wise Best@4 Upper Bound definition is invalid."
        )


def _validate_project_relative_option(value: str, option: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"{option} must be a project-relative path.")


def _markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# STARS Experiment Matrix",
        "",
        "## Effective means over all requested samples",
        "",
        (
            "Failed selections contribute 0 to higher-is-better metrics and "
            "1 to repetition."
        ),
        "",
        (
            "Metric-wise Best@4 Upper Bound selects the best valid C1-C4 "
            "candidate independently for each metric; its values may come "
            "from different candidates and do not represent one deployable output."
        ),
        "",
        "| Dataset | Model | Method | Success | Reward | MV-Align | CSR | SPC | CIDEr-lite | Rep |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        for method, label in (
            ("direct_generation_c1", "Direct Generation (C1)"),
            ("stars_best_of_4", "STARS"),
        ):
            payload = row["methods"][method]
            lines.append(
                _method_markdown_row(
                    row,
                    label,
                    payload["selection_success_rate"],
                    payload["effective_means"],
                )
            )
        upper = row["metric_wise_best_of_4_upper_bound"]
        lines.append(
            _method_markdown_row(
                row,
                "Metric-wise Best@4 Upper Bound",
                upper["selection_success_rate"],
                upper["effective_means"],
            )
        )
    lines.extend(
        [
            "",
            "## Conditional means over successful selections only",
            "",
            "| Dataset | Model | Method | Success | Reward | MV-Align | CSR | SPC | CIDEr-lite | Rep |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        for method, label in (
            ("direct_generation_c1", "Direct Generation (C1)"),
            ("stars_best_of_4", "STARS"),
        ):
            payload = row["methods"][method]
            lines.append(
                _method_markdown_row(
                    row,
                    label,
                    payload["selection_success_rate"],
                    payload["conditional_means"],
                )
            )
        upper = row["metric_wise_best_of_4_upper_bound"]
        lines.append(
            _method_markdown_row(
                row,
                "Metric-wise Best@4 Upper Bound",
                upper["selection_success_rate"],
                upper["conditional_means"],
            )
        )
    lines.extend(
        [
            "",
            "## Mean efficiency per requested sample",
            "",
            "| Dataset | Model | Method | Accounted online latency (s) | Input tokens | Output tokens | Total tokens | Requests |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        efficiency = row["efficiency"]
        lines.append(
            f"| {row['dataset']} | {row['model']} | Direct Generation (C1) | "
            f"{_format_number(efficiency['mean_direct_generation_seconds'])} | "
            f"{_format_number(efficiency['mean_direct_generation_input_tokens'])} | "
            f"{_format_number(efficiency['mean_direct_generation_output_tokens'])} | "
            f"{_format_number(efficiency['mean_direct_generation_total_tokens'])} | "
            f"{_format_number(efficiency['mean_direct_generation_requests'])} |"
        )
        lines.append(
            f"| {row['dataset']} | {row['model']} | STARS | "
            f"{_format_number(efficiency['mean_stars_seconds'])} | "
            f"{_format_number(efficiency['mean_stars_input_tokens'])} | "
            f"{_format_number(efficiency['mean_stars_output_tokens'])} | "
            f"{_format_number(efficiency['mean_stars_total_tokens'])} | "
            f"{_format_number(efficiency['mean_stars_generation_requests'])} |"
        )
    return "\n".join(lines) + "\n"


def _method_markdown_row(
    row: dict[str, Any],
    label: str,
    success_rate: float,
    metrics: dict[str, float | None],
) -> str:
    values = " | ".join(_format_number(metrics[key]) for key in PAPER_METRICS)
    return (
        f"| {row['dataset']} | {row['model']} | {label} | "
        f"{_format_number(success_rate)} | {values} |"
    )


def _format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    main()
