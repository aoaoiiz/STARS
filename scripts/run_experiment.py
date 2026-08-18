from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.checkpoint_identity import (
    LOCAL_CHECKPOINT_KIND,
    validate_checkpoint_identity,
)
from creative_video_exp.config import (
    ExperimentConfig,
    validate_videollama2_runtime_identity,
)
from creative_video_exp.data import load_samples
from creative_video_exp.failure_aware_postprocessing import (
    analyze_failure_aware_candidate_pool,
    write_failure_aware_candidate_pool_analysis,
)
from creative_video_exp.metrics import summarize_results
from creative_video_exp.modeling import (
    build_frame_encoder,
    build_script_generation_model,
)
from creative_video_exp.provenance import (
    build_protocol_manifest,
    prepare_protocol_manifest,
)
from creative_video_exp.reporting import (
    print_metrics_summary,
    write_markdown_report,
)
from creative_video_exp.reward import SelfRewardScorer
from creative_video_exp.semantic_points import (
    FORMAL_REFERENCE_PROTOCOL,
    SemanticPointEvaluator,
    formal_reference_contract_issue,
)
from creative_video_exp.text_metrics import evaluate_generation_quality
from creative_video_exp.utils import (
    ensure_dir,
    iter_jsonl,
    set_seed,
    stable_hash_int,
    write_json,
    write_jsonl,
)
from creative_video_exp.video import SparseFrameSampler


PROTOCOL_NAME = "stars"
PROMPT_VERSION = "stars_visual_only_fixed_validation_retry_v2"
PARSE_RETRY_COUNT = 7
OUTPUT_LANGUAGE = "English"
CANDIDATE_SLOTS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STARS.")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run from results.jsonl.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress after this many newly processed samples.",
    )
    return parser.parse_args()


def _validate_config(config: ExperimentConfig) -> None:
    if config.experiment_version != PROTOCOL_NAME:
        raise RuntimeError(
            f"experiment_version must be `{PROTOCOL_NAME}`."
        )
    generation = config.generation
    required_generation_values = {
        "prompt_version": PROMPT_VERSION,
        "parse_retry_count": PARSE_RETRY_COUNT,
        "output_language": OUTPUT_LANGUAGE,
        "input_protocol": "visual_only",
        "num_candidates": CANDIDATE_SLOTS,
        "candidate_generation_protocol": "independent_single_candidate_calls",
        "pre_score_processing": "json_envelope_and_schema_canonicalization_only",
        "candidate_slot_failure_policy": "retain_invalid_slot_and_continue",
        "method_failure_aggregation": "conditional_and_failure_aware_effective",
    }
    for field, expected in required_generation_values.items():
        observed = getattr(generation, field)
        if observed != expected:
            raise RuntimeError(
                f"generation.{field} must be {expected!r}; received {observed!r}."
            )
    if config.models.mode != "server_full_matrix":
        raise RuntimeError("models.mode must be `server_full_matrix`.")
    generation_endpoint = config.models.get(config.models.active_video_model)
    reward_endpoint = config.models.get(
        config.models.active_reward_vision_model
    )
    if generation_endpoint is None or not generation_endpoint.enabled:
        raise RuntimeError("The active generation endpoint is missing or disabled.")
    if generation_endpoint.provider != "openai_compatible":
        raise RuntimeError(
            "The generation endpoint must use provider `openai_compatible`."
        )
    if reward_endpoint is None or not reward_endpoint.enabled:
        raise RuntimeError("The active reward endpoint is missing or disabled.")
    if reward_endpoint.provider != "huggingface_local":
        raise RuntimeError(
            "The reward endpoint must use provider `huggingface_local`."
        )
    if reward_endpoint.adapter != "siglip2_frame_encoder":
        raise RuntimeError(
            "The reward endpoint must use adapter `siglip2_frame_encoder`."
        )
    for endpoint in (generation_endpoint, reward_endpoint):
        validate_checkpoint_identity(endpoint.checkpoint_identity)
        if endpoint.checkpoint_identity.get("kind") != LOCAL_CHECKPOINT_KIND:
            raise RuntimeError(
                f"Endpoint `{endpoint.id}` must use a local checkpoint identity."
            )
        if endpoint.checkpoint_identity.get("model_id") != endpoint.name:
            raise RuntimeError(
                f"Endpoint `{endpoint.id}` checkpoint identity does not match its model name."
            )
    config.reward.validate_formal_protocol()


def _validate_semantic_points(
    samples: list[Any],
    protocol_manifest: dict[str, Any],
) -> None:
    construction_protocol = (
        protocol_manifest.get("evaluation_definition", {})
        .get("semantic_point_coverage", {})
        .get("reference_construction_protocol", "")
    )
    if construction_protocol != FORMAL_REFERENCE_PROTOCOL:
        raise RuntimeError(
            "The semantic-point reference protocol is missing or invalid."
        )
    missing = [
        sample.video_id
        for sample in samples
        if not sample.semantic_points
        or sample.semantic_point_source_field != "semantic_points"
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} samples lack canonical semantic-point references: "
            + ", ".join(missing[:5])
        )
    invalid = [
        (sample.video_id, formal_reference_contract_issue(sample.raw))
        for sample in samples
        if formal_reference_contract_issue(sample.raw)
    ]
    if invalid:
        preview = "; ".join(
            f"{video_id}: {issue}" for video_id, issue in invalid[:5]
        )
        raise RuntimeError(
            f"{len(invalid)} samples have invalid semantic-point references: {preview}"
        )


def _validate_runtime_reward_identity(
    config: ExperimentConfig,
    encoder_report: dict[str, Any],
) -> None:
    if encoder_report.get("runtime") != "frozen_siglip2_vision_text_reward":
        raise RuntimeError(
            "The frozen SigLIP2 vision-text reward encoder is required."
        )
    endpoint = config.models.get(config.models.active_reward_vision_model)
    if endpoint is None:
        raise RuntimeError("The active reward endpoint cannot be resolved.")
    expected = endpoint.checkpoint_identity
    if encoder_report.get("checkpoint_identity") != expected:
        raise RuntimeError(
            "The loaded reward checkpoint does not match the configured identity."
        )
    if encoder_report.get("checkpoint_identity_sha256") != expected.get(
        "identity_sha256"
    ):
        raise RuntimeError(
            "The loaded reward checkpoint fingerprint does not match the configuration."
        )


def _generation_infrastructure_issue(attempt: dict[str, Any]) -> str:
    if attempt.get("status") == "endpoint_error":
        return "contains an endpoint error"
    if attempt.get("infrastructure_error") is True:
        return "is marked as an infrastructure error"
    transport_attempts = attempt.get("transport_attempts")
    if not isinstance(transport_attempts, list) or not transport_attempts:
        return "has no auditable transport attempts"
    for transport in transport_attempts:
        if not isinstance(transport, dict):
            return "contains a malformed transport attempt"
        status = transport.get("status")
        if status == "error":
            return "contains a failed transport attempt"
        if status != "success":
            return "contains a transport attempt with an invalid status"
    return ""


def _validate_runtime_generation_identity(
    config: ExperimentConfig,
    generation_trace: list[dict[str, Any]],
    expected_visual_frames: int,
) -> None:
    endpoint = config.models.get(config.models.active_video_model)
    if endpoint is None:
        raise RuntimeError("The active generation endpoint cannot be resolved.")
    if not 1 <= expected_visual_frames <= config.sampling.max_frames:
        raise RuntimeError(
            "The sparse observation must contain between one and 16 frames."
        )
    expected = endpoint.checkpoint_identity
    expected_fields = {
        "checkpoint_identity_sha256": expected.get("identity_sha256"),
        "checkpoint_model_id": expected.get("model_id"),
        "checkpoint_revision": expected.get("revision"),
    }
    for slot_position, slot in enumerate(generation_trace, start=1):
        candidate_index = int(slot.get("candidate_index", slot_position))
        attempts = slot.get("attempts", [])
        if not isinstance(attempts, list) or not attempts:
            raise RuntimeError(
                f"Candidate slot C{candidate_index} has no generation trace."
            )
        for attempt_position, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, dict):
                raise RuntimeError(
                    f"C{candidate_index} attempt {attempt_position} is not an object."
                )
            infrastructure_issue = _generation_infrastructure_issue(attempt)
            if infrastructure_issue:
                raise RuntimeError(
                    f"C{candidate_index} attempt {attempt_position} "
                    f"{infrastructure_issue}."
                )
            server_metrics = attempt.get("server_metrics")
            if not isinstance(server_metrics, dict):
                raise RuntimeError(
                    f"C{candidate_index} attempt {attempt_position} lacks model identity metadata."
                )
            if server_metrics.get("server_protocol_version") != PROTOCOL_NAME:
                raise RuntimeError(
                    f"C{candidate_index} attempt {attempt_position} used an unexpected server protocol."
                )
            if server_metrics.get("configured_max_frames") != 16:
                raise RuntimeError(
                    f"C{candidate_index} attempt {attempt_position} used an unexpected service frame cap."
                )
            for field in (
                "requested_visual_input_frames",
                "visual_input_frames",
            ):
                observed_frames = server_metrics.get(field)
                if (
                    isinstance(observed_frames, bool)
                    or not isinstance(observed_frames, int)
                    or observed_frames != expected_visual_frames
                ):
                    raise RuntimeError(
                        f"C{candidate_index} attempt {attempt_position} reports "
                        f"{field}={observed_frames!r}; expected "
                        f"{expected_visual_frames}."
                    )
            if server_metrics.get("returned_model_id_conflict") is True:
                raise RuntimeError(
                    f"C{candidate_index} attempt {attempt_position} reports conflicting model identities."
                )
            if server_metrics.get("returned_model_id") != expected.get("model_id"):
                raise RuntimeError(
                    f"C{candidate_index} attempt {attempt_position} used an unexpected model."
                )
            observed_fields = {
                key: server_metrics.get(key) for key in expected_fields
            }
            if observed_fields != expected_fields:
                raise RuntimeError(
                    f"C{candidate_index} attempt {attempt_position} used an unexpected checkpoint."
                )
            if endpoint.runtime_identity and server_metrics.get(
                "generation_runtime_identity"
            ) != endpoint.runtime_identity:
                raise RuntimeError(
                    f"C{candidate_index} attempt {attempt_position} used an unexpected model runtime."
                )


def _validate_generation_prompt_protocol(
    model_report: dict[str, Any],
    expected_slots: int,
) -> None:
    base_hash = str(model_report.get("base_prompt_sha256", ""))
    retry_hash = str(model_report.get("validation_retry_prompt_sha256", ""))
    trace = list(model_report.get("last_generation_trace", []))
    recorded_hashes = list(model_report.get("prompt_sha256s", []))
    if not base_hash or not retry_hash or base_hash == retry_hash:
        raise RuntimeError("Distinct base and validation-retry prompts are required.")
    if len(trace) != expected_slots:
        raise RuntimeError(
            f"Expected {expected_slots} candidate traces; received {len(trace)}."
        )
    trace_hashes: list[str] = []
    for slot in trace:
        attempts = list(slot.get("attempts", []))
        if not attempts:
            raise RuntimeError("Every candidate slot must contain an attempt trace.")
        for index, attempt in enumerate(attempts):
            expected_kind = (
                "base_generation" if index == 0 else "validation_retry"
            )
            expected_hash = base_hash if index == 0 else retry_hash
            if attempt.get("prompt_kind") != expected_kind:
                raise RuntimeError("A generation attempt used an unexpected prompt kind.")
            if attempt.get("prompt_sha256") != expected_hash:
                raise RuntimeError("A generation attempt used an unexpected prompt.")
            trace_hashes.append(expected_hash)
    if trace_hashes != recorded_hashes:
        raise RuntimeError("Prompt hash accounting does not match the attempt trace.")


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_file(args.config)
    _validate_config(config)
    generation_endpoint = config.models.get(config.models.active_video_model)
    if generation_endpoint is None:
        raise RuntimeError("The active generation endpoint cannot be resolved.")
    set_seed(config.seed)
    output_dir = ensure_dir(PROJECT_ROOT / config.output_dir)
    protocol_manifest = build_protocol_manifest(config, PROJECT_ROOT)
    protocol_fingerprint = str(protocol_manifest["protocol_fingerprint"])
    samples = load_samples(config.data, project_root=PROJECT_ROOT)
    _validate_semantic_points(samples, protocol_manifest)
    expected_sample_keys = _manifest_sample_keys(samples)
    requested_sample_count = len(expected_sample_keys)
    prepare_protocol_manifest(output_dir, protocol_manifest, resume=args.resume)
    results_path = output_dir / "results.jsonl"
    failure_path = output_dir / "generation_failures.jsonl"
    existing_results = (
        _load_existing_results(results_path) if args.resume else {}
    )
    _validate_resume_rows(
        existing_results,
        expected_sample_keys,
        protocol_fingerprint,
        generation_checkpoint_identity=generation_endpoint.checkpoint_identity,
        generation_runtime_identity=generation_endpoint.runtime_identity,
    )
    if not args.resume:
        results_path.write_text("", encoding="utf-8")
        failure_path.write_text("", encoding="utf-8")
    pending_samples = [
        sample
        for sample in samples
        if _sample_key(sample) not in existing_results
    ]

    setup_started = time.perf_counter()
    sampler = SparseFrameSampler(config.sampling)
    encoder_load_started = time.perf_counter()
    encoder, encoder_report = build_frame_encoder(config.models)
    encoder_load_seconds = time.perf_counter() - encoder_load_started
    _validate_runtime_reward_identity(config, encoder_report)
    generation_model = build_script_generation_model(config.models)
    runtime_setup_seconds = time.perf_counter() - setup_started
    scorer = SelfRewardScorer(config.reward, config.generation)
    semantic_point_evaluator = SemanticPointEvaluator(
        encoder,
        config.evaluation,
    )

    results = list(existing_results.values())
    newly_processed = 0
    for sample in pending_samples:
        sample_started = time.perf_counter()
        sample_key = _sample_key(sample)
        sampling_started = time.perf_counter()
        batch = sampler.sample(
            video_path=sample.video_path,
            video_id=sample.video_id,
            metadata={"input_protocol": "visual_only"},
        )
        sampling_seconds = time.perf_counter() - sampling_started
        if batch.source_kind not in {"video", "image_dir", "npz"}:
            raise RuntimeError(
                f"Real video frames could not be loaded for `{sample.video_id}`."
            )
        if batch.metadata.get("fallback_used") is True:
            raise RuntimeError(
                f"Formal temporal decoding fell back for `{sample.video_id}`."
            )
        if batch.metadata.get("formal_sampling_eligible") is not True:
            raise RuntimeError(
                f"Temporal sampling is not formally eligible for `{sample.video_id}`."
            )
        sampled_frame_count = len(batch.frames)
        if (
            not 1 <= sampled_frame_count <= config.sampling.max_frames
            or len(batch.selected_indices) != sampled_frame_count
            or len(batch.bin_ids) != sampled_frame_count
            or batch.selected_indices != sorted(set(batch.selected_indices))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                for index in batch.selected_indices
            )
            or any(
                isinstance(bin_id, bool)
                or not isinstance(bin_id, int)
                or not 0 <= bin_id < config.sampling.num_bins
                for bin_id in batch.bin_ids
            )
        ):
            raise RuntimeError(
                f"Sparse-observation accounting is invalid for `{sample.video_id}`."
            )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        encoding_started = time.perf_counter()
        representation = encoder.encode(batch, context_text="")
        reward_encoding_seconds = time.perf_counter() - encoding_started
        semantic_reference = semantic_point_evaluator.prepare(
            sample.semantic_points,
            source_field=sample.semantic_point_source_field,
        )
        if not semantic_reference.available:
            raise RuntimeError(
                f"SPC reference encoding failed for `{sample.video_id}`."
            )

        generation_started = time.perf_counter()
        candidates = generation_model.generate(sample, batch, config.generation)
        generation_seconds = time.perf_counter() - generation_started
        generation_trace = list(
            generation_model.model_report.get("last_generation_trace", [])
        )
        if len(generation_trace) != CANDIDATE_SLOTS:
            raise RuntimeError(
                f"The generation endpoint returned {len(generation_trace)} slot traces; expected {CANDIDATE_SLOTS}."
            )
        _validate_generation_prompt_protocol(
            generation_model.model_report,
            CANDIDATE_SLOTS,
        )
        _validate_runtime_generation_identity(
            config,
            generation_trace,
            sampled_frame_count,
        )
        server_metrics = list(
            generation_model.model_report.get("last_slot_server_metrics", [])
        )
        reported_metrics = [item for item in server_metrics if item]
        if not reported_metrics or any(
            item.get("server_protocol_version") != PROTOCOL_NAME
            for item in reported_metrics
        ):
            raise RuntimeError(
                "The generation service does not report server_protocol_version=`stars`."
            )
        if len({candidate.candidate_id for candidate in candidates}) != len(
            candidates
        ):
            raise RuntimeError(
                f"Candidate identifiers are not unique for `{sample.video_id}`."
            )
        _validate_candidate_provenance(candidates)

        candidate_rows: list[dict[str, Any]] = []
        seen_candidate_indices: set[int] = set()
        reward_scoring_seconds = 0.0
        metric_evaluation_seconds = 0.0
        reward_seconds_by_slot = [0.0] * CANDIDATE_SLOTS
        for fallback_index, candidate in enumerate(candidates, start=1):
            provenance = candidate.generation_provenance or {}
            candidate_index = int(
                provenance.get("candidate_index", fallback_index)
            )
            if not 1 <= candidate_index <= CANDIDATE_SLOTS:
                raise RuntimeError(
                    f"Candidate `{candidate.candidate_id}` has an invalid slot index."
                )
            if candidate_index in seen_candidate_indices:
                raise RuntimeError(
                    f"Candidate slot C{candidate_index} is duplicated."
                )
            seen_candidate_indices.add(candidate_index)
            reward_started = time.perf_counter()
            reward = scorer.score(candidate, representation)
            reward_seconds = time.perf_counter() - reward_started
            reward_scoring_seconds += reward_seconds
            reward_seconds_by_slot[candidate_index - 1] = reward_seconds
            candidate_row = candidate.as_dict()
            candidate_row.update(
                {
                    "candidate_index": candidate_index,
                    "video_id": sample.video_id,
                    "reward": reward.as_dict(),
                    "generation_source": "multimodal_model",
                }
            )
            metric_started = time.perf_counter()
            candidate_row["quality"] = evaluate_generation_quality(
                candidate,
                sample,
                semantic_point_evaluator=semantic_point_evaluator,
                semantic_point_reference=semantic_reference,
            )
            metric_evaluation_seconds += time.perf_counter() - metric_started
            candidate_rows.append(candidate_row)
        candidate_rows.sort(key=lambda row: int(row["candidate_index"]))
        slot_outcomes = _build_slot_outcomes(
            generation_trace,
            candidate_rows,
            CANDIDATE_SLOTS,
        )
        if len(candidate_rows) < CANDIDATE_SLOTS:
            _record_generation_failure(
                failure_path,
                sample,
                sample_key,
                config.data.name,
                candidate_rows,
                slot_outcomes,
                protocol_fingerprint,
            )

        selection_started = time.perf_counter()
        selected = (
            max(
                candidate_rows,
                key=lambda row: float(row["reward"]["total"]),
            )
            if candidate_rows
            else None
        )
        selection_seconds = time.perf_counter() - selection_started
        baseline = next(
            (
                row
                for row in candidate_rows
                if int(row["candidate_index"]) == 1
            ),
            None,
        )
        slot_seconds = [
            float(slot.get("request_seconds", 0.0)) for slot in slot_outcomes
        ]
        slot_usage = [
            _usage(slot.get("usage", {})) for slot in slot_outcomes
        ]
        prefix_seconds = _prefix_sums(slot_seconds)
        prefix_input_tokens = _prefix_sums(
            usage["input_tokens"] for usage in slot_usage
        )
        prefix_output_tokens = _prefix_sums(
            usage["output_tokens"] for usage in slot_usage
        )
        prefix_total_tokens = _prefix_sums(
            usage["total_tokens"] for usage in slot_usage
        )
        prefix_transport_requests = _prefix_sums(
            int(slot.get("transport_request_count", 0))
            for slot in slot_outcomes
        )
        prefix_semantic_attempts = _prefix_sums(
            int(slot.get("request_count", 0)) for slot in slot_outcomes
        )
        memory = _runner_memory()
        generation_usage = generation_model.model_report.get(
            "last_generation_usage", {}
        )
        result_row = {
            "experiment_version": PROTOCOL_NAME,
            "protocol_fingerprint": protocol_fingerprint,
            "input_protocol": config.generation.input_protocol,
            "prompt_version": config.generation.prompt_version,
            "prompt_sha256": generation_model.model_report.get(
                "last_prompt_sha256", ""
            ),
            "prompt_sha256s": generation_model.model_report.get(
                "prompt_sha256s", []
            ),
            "prompt_templates": generation_model.model_report.get(
                "prompt_templates", {}
            ),
            "candidate_generation_protocol": config.generation.candidate_generation_protocol,
            "pre_score_processing": config.generation.pre_score_processing,
            "sample_key": sample_key,
            "video_id": sample.video_id,
            "dataset": config.data.name,
            "video_path": sample.video_path,
            "evaluation_only_reference": {
                "caption": sample.caption,
                "question": sample.question,
                "answer": sample.answer,
                "category": sample.category,
                "semantic_points": sample.semantic_points,
                "semantic_point_source_field": sample.semantic_point_source_field,
            },
            "semantic_point_evaluation": semantic_reference.audit_dict(),
            "sampling": {
                "source_kind": batch.source_kind,
                "selected_indices": batch.selected_indices,
                "bin_ids": batch.bin_ids,
                **batch.metadata,
            },
            "representation": {
                "content_tags": representation.content_tags,
                "keyframe_source_indices": representation.keyframe_source_indices,
                "aggregation_weights": [
                    round(float(value), 6)
                    for value in representation.aggregation_weights
                ],
                "diagnostics": representation.diagnostics,
            },
            "baseline_candidate_id": (
                baseline["candidate_id"] if baseline is not None else None
            ),
            "stars_candidate_id": (
                selected["candidate_id"] if selected is not None else None
            ),
            "best_candidate": selected,
            "method_output_status": {
                "direct_generation_c1": (
                    "success" if baseline is not None else "failure"
                ),
                "stars_best_of_4": (
                    "success" if selected is not None else "failure"
                ),
            },
            "requested_candidate_slots": CANDIDATE_SLOTS,
            "valid_candidate_slots": len(candidate_rows),
            "failed_candidate_slots": CANDIDATE_SLOTS - len(candidate_rows),
            "full_candidate_pool": len(candidate_rows) == CANDIDATE_SLOTS,
            "candidate_pool_size": len(candidate_rows),
            "selection_rule": "argmax_reward_over_valid_candidates_c1_to_c4",
            "slot_outcomes": slot_outcomes,
            "candidates": candidate_rows,
            "generation_source": "multimodal_model",
            "efficiency": {
                "sampling_seconds": round(sampling_seconds, 6),
                "reward_encoding_seconds": round(
                    reward_encoding_seconds,
                    6,
                ),
                "generation_seconds": round(generation_seconds, 6),
                "generation_requests": int(prefix_transport_requests[-1]),
                "generation_semantic_attempts": int(
                    prefix_semantic_attempts[-1]
                ),
                "generation_input_tokens": int(
                    generation_usage.get(
                        "input_tokens",
                        prefix_input_tokens[-1],
                    )
                ),
                "generation_output_tokens": int(
                    generation_usage.get(
                        "output_tokens",
                        prefix_output_tokens[-1],
                    )
                ),
                "generation_total_tokens": int(
                    generation_usage.get(
                        "total_tokens",
                        prefix_total_tokens[-1],
                    )
                ),
                "reward_scoring_seconds": round(
                    reward_scoring_seconds,
                    6,
                ),
                "metric_evaluation_seconds": round(
                    metric_evaluation_seconds,
                    6,
                ),
                "reward_scoring_seconds_by_slot": [
                    round(value, 6) for value in reward_seconds_by_slot
                ],
                "selection_seconds": round(selection_seconds, 9),
                "direct_generation_seconds": round(
                    sampling_seconds + prefix_seconds[0],
                    6,
                ),
                "stars_seconds": round(
                    sampling_seconds
                    + prefix_seconds[-1]
                    + reward_encoding_seconds
                    + reward_scoring_seconds
                    + selection_seconds,
                    6,
                ),
                "direct_generation_input_tokens": int(
                    prefix_input_tokens[0]
                ),
                "stars_input_tokens": int(prefix_input_tokens[-1]),
                "direct_generation_output_tokens": int(
                    prefix_output_tokens[0]
                ),
                "stars_output_tokens": int(prefix_output_tokens[-1]),
                "direct_generation_total_tokens": int(
                    prefix_total_tokens[0]
                ),
                "stars_total_tokens": int(prefix_total_tokens[-1]),
                "direct_generation_requests": int(
                    prefix_transport_requests[0]
                ),
                "stars_generation_requests": int(
                    prefix_transport_requests[-1]
                ),
                "direct_generation_semantic_attempts": int(
                    prefix_semantic_attempts[0]
                ),
                "stars_semantic_attempts": int(
                    prefix_semantic_attempts[-1]
                ),
                "slot_generation_seconds_including_retries": [
                    round(value, 6) for value in slot_seconds
                ],
                "slot_generation_usage_including_retries": slot_usage,
                "slot_server_metrics": server_metrics,
                "vlm_server_peak_allocated_mib": round(
                    max(
                        [
                            float(item.get("peak_allocated_mib", 0.0))
                            for item in server_metrics
                        ]
                        or [0.0]
                    ),
                    3,
                ),
                "vlm_server_peak_reserved_mib": round(
                    max(
                        [
                            float(item.get("peak_reserved_mib", 0.0))
                            for item in server_metrics
                        ]
                        or [0.0]
                    ),
                    3,
                ),
                "total_pipeline_seconds": round(
                    time.perf_counter() - sample_started,
                    6,
                ),
                "sampled_frames": len(batch.frames),
                **memory,
            },
        }
        results.append(result_row)
        _append_jsonl(results_path, result_row)
        newly_processed += 1
        if (
            args.progress_every > 0
            and newly_processed % args.progress_every == 0
        ):
            print(
                f"Progress: {newly_processed}/{len(pending_samples)} new samples; "
                f"video_id={sample.video_id}; valid_slots={len(candidate_rows)}/{CANDIDATE_SLOTS}.",
                flush=True,
            )

    results = sorted(
        results,
        key=lambda row: (
            str(row.get("video_id", "")),
            str(row.get("sample_key", "")),
        ),
    )
    write_jsonl(results_path, results)
    audit = _run_audit(
        results,
        expected_sample_keys,
        protocol_fingerprint,
        generation_checkpoint_identity=generation_endpoint.checkpoint_identity,
        generation_runtime_identity=generation_endpoint.runtime_identity,
    )
    if audit["evidence_eligible"] is not True:
        raise RuntimeError(
            "The completed result set failed the formal contract audit: "
            f"duplicates={audit['duplicate_result_sample_keys'][:5]}, "
            f"missing={audit['missing_manifest_sample_keys'][:5]}, "
            f"unexpected={audit['unexpected_result_sample_keys'][:5]}, "
            f"row_issues={audit['slot_accounting_issues'][:5]}."
        )
    candidate_summary, candidate_selections = (
        analyze_failure_aware_candidate_pool(
            results=results,
            reward_config=config.reward.__dict__,
            pool_size=CANDIDATE_SLOTS,
            bootstrap_replicates=1000,
            seed=config.seed,
        )
    )
    analysis_dir = output_dir / "candidate_pool_analysis"
    write_failure_aware_candidate_pool_analysis(
        analysis_dir,
        candidate_summary,
        candidate_selections,
    )
    metrics = summarize_results(results)
    metrics.update(audit)
    metrics["model_runtime"] = {
        "video_model": {
            **encoder_report,
            "load_seconds": round(encoder_load_seconds, 6),
        },
        "script_generation_model": generation_model.model_report,
        "runner_setup_seconds": round(runtime_setup_seconds, 6),
    }
    metrics["config"] = config.as_dict()
    metrics["protocol"] = protocol_manifest
    metrics["runtime_environment"] = _runtime_environment()
    metrics["candidate_pool_analysis_path"] = str(
        analysis_dir / "failure_aware_candidate_pool_analysis.json"
    )
    metrics["method_output_success_rates"] = {
        method: payload["selection_success_rate"]
        for method, payload in candidate_summary["methods"].items()
    }
    metrics["method_conditional_means"] = {
        method: payload["conditional_means"]
        for method, payload in candidate_summary["methods"].items()
    }
    metrics["method_effective_means"] = {
        method: payload["effective_means"]
        for method, payload in candidate_summary["methods"].items()
    }
    metrics["metric_wise_best_of_4_upper_bound"] = candidate_summary[
        "metric_wise_best_of_4_upper_bound"
    ]
    metrics["metric_wise_best_of_4_upper_bound_definition"] = (
        candidate_summary[
            "metric_wise_best_of_4_upper_bound_definition"
        ]
    )
    write_json(output_dir / "metrics.json", metrics)
    write_markdown_report(output_dir / "metrics_report.md", metrics)
    run_status = {
        "experiment_version": PROTOCOL_NAME,
        "protocol_fingerprint": protocol_fingerprint,
        "status": "eligible" if audit["evidence_eligible"] else "incomplete",
        **audit,
    }
    write_json(output_dir / "run_status.json", run_status)
    print(f"Requested samples: {requested_sample_count}")
    print(f"Completed samples: {len(results)}")
    print(f"Candidates: {sum(len(row['candidates']) for row in results)}")
    print(f"Evidence eligible: {audit['evidence_eligible']}")
    print(f"Outputs: {output_dir}")
    print_metrics_summary(metrics)


def _validate_candidate_provenance(candidates: list[Any]) -> None:
    for candidate in candidates:
        provenance = candidate.generation_provenance or {}
        if (
            provenance.get("parse_status") != "valid_json_candidate"
            or provenance.get("semantic_repair_applied") is not False
            or provenance.get("wrapped_free_text") is not False
            or provenance.get("surrounding_free_text") is not False
            or provenance.get("response_envelope")
            not in {"raw_json", "single_markdown_json_fence"}
        ):
            raise RuntimeError(
                f"Candidate `{candidate.candidate_id}` failed structured validation."
            )


def _build_slot_outcomes(
    generation_trace: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    expected_slots: int,
) -> list[dict[str, Any]]:
    if len(generation_trace) != expected_slots:
        raise RuntimeError(
            f"Expected {expected_slots} slot traces; received {len(generation_trace)}."
        )
    candidates_by_index = {
        int(candidate["candidate_index"]): candidate
        for candidate in candidate_rows
    }
    outcomes: list[dict[str, Any]] = []
    for expected_index, source in enumerate(generation_trace, start=1):
        candidate_index = int(source.get("candidate_index", 0))
        if candidate_index != expected_index:
            raise RuntimeError("Candidate traces must be ordered C1-C4.")
        candidate = candidates_by_index.get(candidate_index)
        accepted = bool(source.get("accepted"))
        if accepted != (candidate is not None):
            raise RuntimeError(
                f"C{candidate_index} acceptance and scored candidate disagree."
            )
        attempts = list(source.get("attempts", []))
        if not attempts:
            raise RuntimeError(
                f"C{candidate_index} has no auditable generation attempts."
            )
        usage = source.get("usage") or {
            key: sum(
                int(attempt.get("usage", {}).get(key, 0))
                for attempt in attempts
            )
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        usage = _usage(usage)
        outcome = dict(source)
        outcome.update(
            {
                "candidate_index": candidate_index,
                "terminal_status": "valid" if candidate is not None else "failed",
                "terminal_reason": str(
                    source.get("terminal_reason")
                    or (
                        "accepted_valid_candidate"
                        if candidate is not None
                        else "bounded_attempts_exhausted"
                    )
                ),
                "candidate_id": (
                    candidate["candidate_id"] if candidate is not None else None
                ),
                "request_count": int(
                    source.get("request_count", len(attempts))
                ),
                "transport_request_count": int(
                    source.get(
                        "transport_request_count",
                        sum(
                            int(
                                attempt.get(
                                    "transport_request_count",
                                    1,
                                )
                            )
                            for attempt in attempts
                        ),
                    )
                ),
                "request_seconds": round(
                    float(
                        source.get(
                            "request_seconds",
                            sum(
                                float(attempt.get("request_seconds", 0.0))
                                for attempt in attempts
                            ),
                        )
                    ),
                    6,
                ),
                "usage": usage,
                "attempts": attempts,
            }
        )
        outcomes.append(outcome)
    return outcomes


def _record_generation_failure(
    failure_path: Path,
    sample: Any,
    sample_key: str,
    dataset: str,
    candidate_rows: list[dict[str, Any]],
    slot_outcomes: list[dict[str, Any]],
    protocol_fingerprint: str,
) -> None:
    record = {
        "experiment_version": PROTOCOL_NAME,
        "protocol_fingerprint": protocol_fingerprint,
        "dataset": dataset,
        "sample_key": sample_key,
        "video_id": sample.video_id,
        "video_path": sample.video_path,
        "expected_candidates": CANDIDATE_SLOTS,
        "valid_candidates": len(candidate_rows),
        "candidate_ids": [row["candidate_id"] for row in candidate_rows],
        "failed_candidate_slot_indices": [
            int(slot["candidate_index"])
            for slot in slot_outcomes
            if slot["terminal_status"] == "failed"
        ],
        "slot_outcomes": slot_outcomes,
    }
    _append_jsonl(failure_path, record)


def _run_audit(
    results: list[dict[str, Any]],
    expected_sample_keys: list[str],
    protocol_fingerprint: str,
    *,
    generation_checkpoint_identity: dict[str, Any] | None = None,
    generation_runtime_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = set(expected_sample_keys)
    observed_keys = [str(result.get("sample_key", "")) for result in results]
    observed = set(observed_keys)
    duplicate_keys = sorted(
        key for key in observed if observed_keys.count(key) > 1
    )
    missing_keys = sorted(expected - observed)
    unexpected_keys = sorted(observed - expected)
    row_issues: list[dict[str, str]] = []
    valid_slots = 0
    failed_slots = 0
    full_pools = 0
    for result in results:
        issue = _result_contract_issue(
            result,
            protocol_fingerprint,
            generation_checkpoint_identity=generation_checkpoint_identity,
            generation_runtime_identity=generation_runtime_identity,
        )
        if issue:
            row_issues.append(
                {
                    "sample_key": str(result.get("sample_key", "")),
                    "issue": issue,
                }
            )
        slots = result.get("slot_outcomes", [])
        valid = sum(slot.get("terminal_status") == "valid" for slot in slots)
        failed = sum(slot.get("terminal_status") == "failed" for slot in slots)
        valid_slots += valid
        failed_slots += failed
        full_pools += int(valid == CANDIDATE_SLOTS)
    requested_slots = len(expected_sample_keys) * CANDIDATE_SLOTS
    eligible = (
        len(results) == len(expected_sample_keys)
        and not duplicate_keys
        and not missing_keys
        and not unexpected_keys
        and not row_issues
        and valid_slots + failed_slots == requested_slots
    )
    return {
        "requested_num_samples": len(expected_sample_keys),
        "completed_num_samples": len(results),
        "requested_candidate_slot_count": requested_slots,
        "valid_candidate_slot_count": valid_slots,
        "failed_candidate_slot_count": failed_slots,
        "candidate_slot_success_rate": round(
            valid_slots / max(1, requested_slots),
            6,
        ),
        "full_candidate_pool_sample_count": full_pools,
        "full_candidate_pool_sample_rate": round(
            full_pools / max(1, len(expected_sample_keys)),
            6,
        ),
        "unresolved_generation_failure_count": len(missing_keys)
        + len(row_issues),
        "duplicate_result_sample_keys": duplicate_keys,
        "missing_manifest_sample_keys": missing_keys,
        "unexpected_result_sample_keys": unexpected_keys,
        "slot_accounting_issues": row_issues,
        "evidence_eligible": eligible,
    }


def _result_contract_issue(
    result: dict[str, Any],
    protocol_fingerprint: str,
    *,
    generation_checkpoint_identity: dict[str, Any] | None = None,
    generation_runtime_identity: dict[str, Any] | None = None,
) -> str:
    if result.get("experiment_version") != PROTOCOL_NAME:
        return "experiment_version is invalid"
    if result.get("protocol_fingerprint") != protocol_fingerprint:
        return "protocol_fingerprint is invalid"
    if result.get("prompt_version") != PROMPT_VERSION:
        return "prompt_version is invalid"
    if result.get("generation_source") != "multimodal_model":
        return "generation source is not a multimodal model"
    sampling = result.get("sampling")
    if not isinstance(sampling, dict):
        return "sampling metadata is missing"
    if sampling.get("source_kind") not in {"video", "image_dir", "npz"}:
        return "video sampling did not use a supported real source"
    if sampling.get("fallback_used") is True:
        return "video sampling used a temporal decoding fallback"
    if sampling.get("formal_sampling_eligible") is not True:
        return "video sampling is not formally eligible"
    selected_indices = sampling.get("selected_indices")
    bin_ids = sampling.get("bin_ids")
    if (
        not isinstance(selected_indices, list)
        or not selected_indices
        or len(selected_indices) > 16
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in selected_indices
        )
        or selected_indices != sorted(set(selected_indices))
    ):
        return "sampled-frame indices are invalid"
    if (
        not isinstance(bin_ids, list)
        or len(bin_ids) != len(selected_indices)
        or any(
            isinstance(bin_id, bool)
            or not isinstance(bin_id, int)
            or not 0 <= bin_id < 8
            for bin_id in bin_ids
        )
    ):
        return "sampled-frame bin accounting is invalid"
    expected_visual_frames = len(selected_indices)
    if int(result.get("requested_candidate_slots", 0)) != CANDIDATE_SLOTS:
        return "requested slot count is invalid"
    slots = result.get("slot_outcomes")
    if not isinstance(slots, list) or len(slots) != CANDIDATE_SLOTS:
        return "slot outcomes are incomplete"
    candidates = result.get("candidates", [])
    if not isinstance(candidates, list):
        return "candidates is not a list"
    if any(not isinstance(candidate, dict) for candidate in candidates):
        return "a scored candidate is not an object"
    try:
        candidates_by_index = {
            int(candidate.get("candidate_index", 0)): candidate
            for candidate in candidates
        }
    except (TypeError, ValueError):
        return "a candidate index is invalid"
    if len(candidates_by_index) != len(candidates):
        return "candidate indices are duplicated"
    valid_candidate_indices: set[int] = set()
    for expected_index, slot in enumerate(slots, start=1):
        try:
            candidate_index = int(slot.get("candidate_index", 0))
        except (TypeError, ValueError):
            return "a slot candidate index is invalid"
        if candidate_index != expected_index:
            return "slot outcomes are not ordered C1-C4"
        status = slot.get("terminal_status")
        if status not in {"valid", "failed"}:
            return "a slot has an invalid terminal status"
        if (status == "valid") != (expected_index in candidates_by_index):
            return "valid slots and scored candidates do not match"
        expected_candidate = candidates_by_index.get(expected_index)
        expected_candidate_id = (
            expected_candidate.get("candidate_id")
            if expected_candidate is not None
            else None
        )
        if slot.get("candidate_id") != expected_candidate_id:
            return "slot and scored-candidate identifiers do not match"
        if status == "valid":
            valid_candidate_indices.add(expected_index)
        attempts = slot.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            return "a slot has no auditable attempts"
        if len(attempts) > PARSE_RETRY_COUNT + 1:
            return "a slot exceeds the bounded semantic-attempt count"
        accepted_attempts = 0
        for attempt_position, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, dict):
                return "a generation attempt is not an object"
            if attempt.get("attempt") != attempt_position:
                return "generation attempts are not consecutively numbered"
            expected_prompt_kind = (
                "base_generation"
                if attempt_position == 1
                else "validation_retry"
            )
            if attempt.get("prompt_kind") != expected_prompt_kind:
                return "a generation attempt used an invalid prompt kind"
            attempt_status = attempt.get("status")
            if attempt_status not in {"accepted", "parse_rejected", "endpoint_error"}:
                return "a generation attempt has an invalid status"
            accepted_attempts += int(attempt_status == "accepted")
            infrastructure_issue = _generation_infrastructure_issue(attempt)
            if infrastructure_issue:
                return f"a generation attempt {infrastructure_issue}"
            server_metrics = attempt.get("server_metrics")
            if not isinstance(server_metrics, dict):
                return "a successful server response lacks server metrics"
            if server_metrics.get("server_protocol_version") != PROTOCOL_NAME:
                return "a successful server response used an invalid protocol"
            if server_metrics.get("configured_max_frames") != 16:
                return "a successful server response used an invalid service frame cap"
            for field in (
                "requested_visual_input_frames",
                "visual_input_frames",
            ):
                observed_frames = server_metrics.get(field)
                if (
                    isinstance(observed_frames, bool)
                    or not isinstance(observed_frames, int)
                    or observed_frames != expected_visual_frames
                ):
                    return (
                        "a successful server response used a mismatched "
                        "visual-frame count"
                    )
            if server_metrics.get("returned_model_id_conflict") is True:
                return "a successful server response has conflicting model identities"
            if generation_checkpoint_identity:
                expected_fields = {
                    "checkpoint_identity_sha256": generation_checkpoint_identity.get(
                        "identity_sha256"
                    ),
                    "checkpoint_model_id": generation_checkpoint_identity.get(
                        "model_id"
                    ),
                    "checkpoint_revision": generation_checkpoint_identity.get(
                        "revision"
                    ),
                }
                if {
                    key: server_metrics.get(key) for key in expected_fields
                } != expected_fields:
                    return "a successful server response used an unexpected checkpoint"
                if server_metrics.get("returned_model_id") != expected_fields[
                    "checkpoint_model_id"
                ]:
                    return "a successful server response used an unexpected model"
            if generation_runtime_identity and server_metrics.get(
                "generation_runtime_identity"
            ) != generation_runtime_identity:
                return "a successful server response used an unexpected model runtime"
            if server_metrics.get("checkpoint_model_id") == "VideoLLaMA2-7B-16F":
                runtime_identity = server_metrics.get(
                    "generation_runtime_identity"
                )
                try:
                    validate_videollama2_runtime_identity(runtime_identity)
                except ValueError as exc:
                    return str(exc)
        if (status == "valid") != (accepted_attempts == 1):
            return "slot status and accepted semantic attempt disagree"
        if accepted_attempts and attempts[-1].get("status") != "accepted":
            return "an accepted semantic attempt is not terminal"
        try:
            accounting_issue = _slot_accounting_issue(slot, attempts, expected_index)
        except RuntimeError as exc:
            return str(exc)
        if accounting_issue:
            return accounting_issue
    if set(candidates_by_index) != valid_candidate_indices:
        return "scored candidate indices do not match valid slots"
    try:
        efficiency_issue = _efficiency_accounting_issue(
            result,
            slots,
            expected_visual_frames,
        )
    except RuntimeError as exc:
        return str(exc)
    if efficiency_issue:
        return efficiency_issue
    valid_count = len(valid_candidate_indices)
    aggregate_expectations = {
        "valid_candidate_slots": valid_count,
        "failed_candidate_slots": CANDIDATE_SLOTS - valid_count,
        "candidate_pool_size": valid_count,
    }
    for key, expected_value in aggregate_expectations.items():
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected_value:
            return f"{key} does not match the terminal slot records"
    if result.get("full_candidate_pool") is not (valid_count == CANDIDATE_SLOTS):
        return "full_candidate_pool does not match the terminal slot records"
    if result.get("selection_rule") != "argmax_reward_over_valid_candidates_c1_to_c4":
        return "selection_rule is invalid"
    baseline = candidates_by_index.get(1)
    baseline_id = baseline.get("candidate_id") if baseline else None
    if result.get("baseline_candidate_id") != baseline_id:
        return "Direct Generation does not match C1"
    candidate_ids = [candidate.get("candidate_id") for candidate in candidates]
    if any(
        not isinstance(candidate_id, str) or not candidate_id
        for candidate_id in candidate_ids
    ):
        return "a scored candidate has no candidate identifier"
    if len(candidate_ids) != len(set(candidate_ids)):
        return "candidate identifiers are duplicated"
    if candidates:
        try:
            ordered_candidates = [
                candidates_by_index[index] for index in sorted(candidates_by_index)
            ]
            reward_totals = [
                _required_candidate_reward(candidate)
                for candidate in ordered_candidates
            ]
        except RuntimeError as exc:
            return str(exc)
        selected_index = max(
            range(len(ordered_candidates)),
            key=reward_totals.__getitem__,
        )
        selected = ordered_candidates[selected_index]
        selected_id = selected["candidate_id"]
    else:
        selected = None
        selected_id = None
    stars_candidate_id = result.get("stars_candidate_id")
    if stars_candidate_id != selected_id:
        return "STARS is not argmax Reward over valid C1-C4 candidates"
    best_candidate = result.get("best_candidate")
    if selected is None:
        if best_candidate is not None:
            return "best_candidate must be null when no valid candidate exists"
    elif not isinstance(best_candidate, dict):
        return "best_candidate is missing for the STARS selection"
    elif best_candidate.get("candidate_id") != stars_candidate_id:
        return "best_candidate does not match stars_candidate_id"
    elif best_candidate != selected:
        return "best_candidate does not equal the Reward-argmax candidate record"
    method_status = result.get("method_output_status")
    if not isinstance(method_status, dict):
        return "method_output_status is missing"
    expected_method_status = {
        "direct_generation_c1": "success" if baseline is not None else "failure",
        "stars_best_of_4": "success" if selected is not None else "failure",
    }
    if method_status != expected_method_status:
        return "method_output_status does not match candidate availability"
    return ""


def _slot_accounting_issue(
    slot: dict[str, Any],
    attempts: list[dict[str, Any]],
    candidate_index: int,
) -> str:
    context = f"slot C{candidate_index}"
    request_count = slot.get("request_count")
    if (
        isinstance(request_count, bool)
        or not isinstance(request_count, int)
        or request_count != len(attempts)
    ):
        return f"{context} request_count does not match its attempts"
    attempt_seconds: list[float] = []
    attempt_usages: list[dict[str, int]] = []
    expected_transport_count = 0
    for position, attempt in enumerate(attempts, start=1):
        attempt_context = f"{context} attempt {position}"
        attempt_seconds.append(
            _nonnegative_number(
                attempt.get("request_seconds"),
                f"{attempt_context}.request_seconds",
            )
        )
        attempt_usage = _usage(attempt.get("usage", {}))
        attempt_usages.append(attempt_usage)
        transports = attempt.get("transport_attempts")
        if not isinstance(transports, list) or not transports:
            return f"{attempt_context} has no transport attempts"
        transport_count = attempt.get("transport_request_count")
        if (
            isinstance(transport_count, bool)
            or not isinstance(transport_count, int)
            or transport_count != len(transports)
        ):
            return f"{attempt_context} transport count is inconsistent"
        expected_transport_count += transport_count
        transport_seconds = 0.0
        transport_backoff = 0.0
        transport_usage = {key: 0 for key in ("input_tokens", "output_tokens", "total_tokens")}
        for transport_position, transport in enumerate(transports, start=1):
            if not isinstance(transport, dict):
                return f"{attempt_context} contains a malformed transport attempt"
            if transport.get("transport_attempt") != transport_position:
                return f"{attempt_context} transport attempts are not consecutively numbered"
            transport_seconds += _nonnegative_number(
                transport.get("request_seconds"),
                f"{attempt_context}.transport_request_seconds",
            )
            transport_backoff += _nonnegative_number(
                transport.get("backoff_seconds_after", 0.0),
                f"{attempt_context}.transport_backoff_seconds",
            )
            usage = _usage(transport.get("usage", {}))
            for key in transport_usage:
                transport_usage[key] += usage[key]
        if not _accounting_close(
            _nonnegative_number(
                attempt.get("transport_request_seconds"),
                f"{attempt_context}.transport_request_seconds",
            ),
            transport_seconds,
        ):
            return f"{attempt_context} transport seconds are inconsistent"
        if not _accounting_close(
            _nonnegative_number(
                attempt.get("transport_backoff_seconds"),
                f"{attempt_context}.transport_backoff_seconds",
            ),
            transport_backoff,
        ):
            return f"{attempt_context} transport backoff is inconsistent"
        if transport_usage != attempt_usage:
            return f"{attempt_context} token usage is inconsistent"
        if attempt_seconds[-1] + 5e-6 < transport_seconds + transport_backoff:
            return f"{attempt_context} request time is shorter than transport time"
    slot_transport_count = slot.get("transport_request_count")
    if (
        isinstance(slot_transport_count, bool)
        or not isinstance(slot_transport_count, int)
        or slot_transport_count != expected_transport_count
    ):
        return f"{context} transport count does not match its attempts"
    if not _accounting_close(
        _nonnegative_number(
            slot.get("request_seconds"),
            f"{context}.request_seconds",
        ),
        sum(attempt_seconds),
    ):
        return f"{context} request seconds do not match its attempts"
    expected_usage = {
        key: sum(usage[key] for usage in attempt_usages)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    if _usage(slot.get("usage", {})) != expected_usage:
        return f"{context} token usage does not match its attempts"
    return ""


def _efficiency_accounting_issue(
    result: dict[str, Any],
    slots: list[dict[str, Any]],
    sampled_frame_count: int,
) -> str:
    efficiency = result.get("efficiency")
    if not isinstance(efficiency, dict):
        return "efficiency accounting is missing"
    slot_usages = [_usage(slot.get("usage", {})) for slot in slots]
    slot_seconds = [
        _nonnegative_number(
            slot.get("request_seconds"),
            f"slot C{index}.request_seconds",
        )
        for index, slot in enumerate(slots, start=1)
    ]
    slot_requests = [
        _required_nonnegative_int(
            slot.get("transport_request_count"),
            f"slot C{index}.transport_request_count",
        )
        for index, slot in enumerate(slots, start=1)
    ]
    slot_semantic_attempts = [
        _required_nonnegative_int(
            slot.get("request_count"),
            f"slot C{index}.request_count",
        )
        for index, slot in enumerate(slots, start=1)
    ]
    total_usage = {
        key: sum(usage[key] for usage in slot_usages)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    integer_expectations = {
        "generation_requests": sum(slot_requests),
        "generation_semantic_attempts": sum(slot_semantic_attempts),
        "generation_input_tokens": total_usage["input_tokens"],
        "generation_output_tokens": total_usage["output_tokens"],
        "generation_total_tokens": total_usage["total_tokens"],
        "direct_generation_input_tokens": slot_usages[0]["input_tokens"],
        "direct_generation_output_tokens": slot_usages[0]["output_tokens"],
        "direct_generation_total_tokens": slot_usages[0]["total_tokens"],
        "stars_input_tokens": total_usage["input_tokens"],
        "stars_output_tokens": total_usage["output_tokens"],
        "stars_total_tokens": total_usage["total_tokens"],
        "direct_generation_requests": slot_requests[0],
        "stars_generation_requests": sum(slot_requests),
        "direct_generation_semantic_attempts": slot_semantic_attempts[0],
        "stars_semantic_attempts": sum(slot_semantic_attempts),
        "sampled_frames": sampled_frame_count,
    }
    for key, expected in integer_expectations.items():
        observed = efficiency.get(key)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed != expected
        ):
            return f"efficiency.{key} is inconsistent with the slot records"
    for prefix in ("generation", "direct_generation", "stars"):
        input_tokens = efficiency[f"{prefix}_input_tokens"]
        output_tokens = efficiency[f"{prefix}_output_tokens"]
        total_tokens = efficiency[f"{prefix}_total_tokens"]
        if total_tokens != input_tokens + output_tokens:
            return f"efficiency.{prefix}_total_tokens is inconsistent"
    sampling_seconds = _nonnegative_number(
        efficiency.get("sampling_seconds"),
        "efficiency.sampling_seconds",
    )
    reward_encoding_seconds = _nonnegative_number(
        efficiency.get("reward_encoding_seconds"),
        "efficiency.reward_encoding_seconds",
    )
    reward_scoring_seconds = _nonnegative_number(
        efficiency.get("reward_scoring_seconds"),
        "efficiency.reward_scoring_seconds",
    )
    selection_seconds = _nonnegative_number(
        efficiency.get("selection_seconds"),
        "efficiency.selection_seconds",
    )
    direct_seconds = _nonnegative_number(
        efficiency.get("direct_generation_seconds"),
        "efficiency.direct_generation_seconds",
    )
    stars_seconds = _nonnegative_number(
        efficiency.get("stars_seconds"),
        "efficiency.stars_seconds",
    )
    expected_direct_seconds = sampling_seconds + slot_seconds[0]
    expected_stars_seconds = (
        sampling_seconds
        + sum(slot_seconds)
        + reward_encoding_seconds
        + reward_scoring_seconds
        + selection_seconds
    )
    if not _accounting_close(direct_seconds, expected_direct_seconds):
        return "efficiency.direct_generation_seconds is inconsistent"
    if not _accounting_close(stars_seconds, expected_stars_seconds):
        return "efficiency.stars_seconds is inconsistent"
    generation_seconds = _nonnegative_number(
        efficiency.get("generation_seconds"),
        "efficiency.generation_seconds",
    )
    if generation_seconds + 5e-6 < sum(slot_seconds):
        return "efficiency.generation_seconds is shorter than request time"
    metric_seconds = _nonnegative_number(
        efficiency.get("metric_evaluation_seconds"),
        "efficiency.metric_evaluation_seconds",
    )
    total_pipeline_seconds = _nonnegative_number(
        efficiency.get("total_pipeline_seconds"),
        "efficiency.total_pipeline_seconds",
    )
    minimum_pipeline_seconds = (
        sampling_seconds
        + generation_seconds
        + reward_encoding_seconds
        + reward_scoring_seconds
        + metric_seconds
        + selection_seconds
    )
    if total_pipeline_seconds + 5e-6 < minimum_pipeline_seconds:
        return "efficiency.total_pipeline_seconds is inconsistent"
    recorded_slot_seconds = efficiency.get(
        "slot_generation_seconds_including_retries"
    )
    if (
        not isinstance(recorded_slot_seconds, list)
        or len(recorded_slot_seconds) != CANDIDATE_SLOTS
        or any(
            not _accounting_close(
                _nonnegative_number(value, "efficiency.slot_generation_seconds"),
                expected,
            )
            for value, expected in zip(recorded_slot_seconds, slot_seconds)
        )
    ):
        return "efficiency slot-generation seconds are inconsistent"
    recorded_slot_usage = efficiency.get(
        "slot_generation_usage_including_retries"
    )
    if (
        not isinstance(recorded_slot_usage, list)
        or len(recorded_slot_usage) != CANDIDATE_SLOTS
        or [_usage(value) for value in recorded_slot_usage] != slot_usages
    ):
        return "efficiency slot-generation usage is inconsistent"
    scoring_by_slot = efficiency.get("reward_scoring_seconds_by_slot")
    if (
        not isinstance(scoring_by_slot, list)
        or len(scoring_by_slot) != CANDIDATE_SLOTS
    ):
        return "efficiency reward-scoring slot accounting is invalid"
    scoring_values = [
        _nonnegative_number(value, "efficiency.reward_scoring_seconds_by_slot")
        for value in scoring_by_slot
    ]
    if not _accounting_close(sum(scoring_values), reward_scoring_seconds):
        return "efficiency reward-scoring seconds are inconsistent"
    return ""


def _nonnegative_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{context} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise RuntimeError(f"{context} must be finite and non-negative.")
    return number


def _required_nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{context} must be a non-negative integer.")
    return value


def _accounting_close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 5e-6


def _required_candidate_reward(candidate: dict[str, Any]) -> float:
    reward = candidate.get("reward")
    if not isinstance(reward, dict) or "total" not in reward:
        raise RuntimeError("a scored candidate lacks reward.total")
    value = reward["total"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("a scored candidate has a non-numeric reward.total")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("a scored candidate has a non-finite reward.total")
    if not 0.0 <= number <= 1.0:
        raise RuntimeError("a scored candidate has reward.total outside [0, 1]")
    return number


def _validate_resume_rows(
    existing_results: dict[str, dict[str, Any]],
    expected_sample_keys: list[str],
    protocol_fingerprint: str,
    *,
    generation_checkpoint_identity: dict[str, Any] | None = None,
    generation_runtime_identity: dict[str, Any] | None = None,
) -> None:
    unexpected = sorted(set(existing_results) - set(expected_sample_keys))
    mismatched = sorted(
        key
        for key, row in existing_results.items()
        if row.get("protocol_fingerprint") != protocol_fingerprint
    )
    contract_issues = {
        key: issue
        for key, row in existing_results.items()
        if (
            issue := _result_contract_issue(
                row,
                protocol_fingerprint,
                generation_checkpoint_identity=generation_checkpoint_identity,
                generation_runtime_identity=generation_runtime_identity,
            )
        )
    }
    if unexpected or mismatched or contract_issues:
        raise RuntimeError(
            "Resume rows do not match the requested protocol: "
            f"unexpected={unexpected[:5]}, mismatched={mismatched[:5]}, "
            f"contract_issues={list(contract_issues.items())[:5]}."
        )


def _load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        sample_key = str(row.get("sample_key") or _sample_key_from_row(row))
        if sample_key in rows:
            raise RuntimeError(f"Duplicate sample key in results: `{sample_key}`.")
        row["sample_key"] = sample_key
        rows[sample_key] = row
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _manifest_sample_keys(samples: list[Any]) -> list[str]:
    keys = [_sample_key(sample) for sample in samples]
    if not keys or len(keys) != len(set(keys)) or any(not key for key in keys):
        raise RuntimeError("The dataset must contain unique, non-empty sample keys.")
    return keys


def _sample_key(sample: Any) -> str:
    return _sample_key_payload(
        {
            "video_id": sample.video_id,
            "question": sample.question,
            "answer": sample.answer,
            "caption": sample.caption,
        }
    )


def _sample_key_from_row(row: dict[str, Any]) -> str:
    reference = row.get("evaluation_only_reference", {})
    return _sample_key_payload(
        {
            "video_id": row.get("video_id", ""),
            "question": reference.get("question", ""),
            "answer": reference.get("answer", ""),
            "caption": reference.get("caption", ""),
        }
    )


def _sample_key_payload(payload: dict[str, Any]) -> str:
    video_id = str(payload.get("video_id", ""))
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"{video_id}::{stable_hash_int(text, 10**12):012d}"


def _prefix_sums(values: Any) -> list[float]:
    total = 0.0
    output: list[float] = []
    for value in values:
        total += float(value)
        output.append(total)
    return output


def _usage(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise RuntimeError("Slot usage must be an object.")
    usage: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                "Slot usage must include integer input, output, and total tokens."
            )
        usage[key] = value
    if any(value < 0 for value in usage.values()):
        raise RuntimeError("Token counts must be non-negative.")
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise RuntimeError("Total tokens must equal input plus output tokens.")
    return usage


def _runner_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "reward_runner_peak_allocated_mib": 0.0,
            "reward_runner_peak_reserved_mib": 0.0,
        }
    return {
        "reward_runner_peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / 1024**2,
            3,
        ),
        "reward_runner_peak_reserved_mib": round(
            torch.cuda.max_memory_reserved() / 1024**2,
            3,
        ),
    }


def _runtime_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "torch", "torchvision", "transformers", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_cuda_version": torch.version.cuda or "none",
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": gpu,
        "packages": packages,
        "token_accounting": "text tokens only; visual tokens are excluded",
    }


if __name__ == "__main__":
    main()
