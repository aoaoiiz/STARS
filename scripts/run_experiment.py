from __future__ import annotations

import argparse
import importlib.metadata
import json
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
from creative_video_exp.config import ExperimentConfig
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


def _validate_runtime_generation_identity(
    config: ExperimentConfig,
    generation_trace: list[dict[str, Any]],
) -> None:
    endpoint = config.models.get(config.models.active_video_model)
    if endpoint is None:
        raise RuntimeError("The active generation endpoint cannot be resolved.")
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
            if attempt.get("status") == "endpoint_error":
                continue
            server_metrics = attempt.get("server_metrics")
            if not isinstance(server_metrics, dict):
                raise RuntimeError(
                    f"C{candidate_index} attempt {attempt_position} lacks model identity metadata."
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
        if batch.source_kind == "synthetic":
            raise RuntimeError(
                f"Real video frames could not be loaded for `{sample.video_id}`."
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
        _validate_runtime_generation_identity(config, generation_trace)
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
        issue = _result_contract_issue(result, protocol_fingerprint)
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
) -> str:
    if result.get("experiment_version") != PROTOCOL_NAME:
        return "experiment_version is invalid"
    if result.get("protocol_fingerprint") != protocol_fingerprint:
        return "protocol_fingerprint is invalid"
    if result.get("prompt_version") != PROMPT_VERSION:
        return "prompt_version is invalid"
    if result.get("generation_source") != "multimodal_model":
        return "generation source is not a multimodal model"
    if result.get("sampling", {}).get("source_kind") == "synthetic":
        return "video sampling used synthetic frames"
    if int(result.get("requested_candidate_slots", 0)) != CANDIDATE_SLOTS:
        return "requested slot count is invalid"
    slots = result.get("slot_outcomes")
    if not isinstance(slots, list) or len(slots) != CANDIDATE_SLOTS:
        return "slot outcomes are incomplete"
    candidates = result.get("candidates", [])
    candidates_by_index = {
        int(candidate.get("candidate_index", 0)): candidate
        for candidate in candidates
    }
    if len(candidates_by_index) != len(candidates):
        return "candidate indices are duplicated"
    for expected_index, slot in enumerate(slots, start=1):
        if int(slot.get("candidate_index", 0)) != expected_index:
            return "slot outcomes are not ordered C1-C4"
        status = slot.get("terminal_status")
        if status not in {"valid", "failed"}:
            return "a slot has an invalid terminal status"
        if (status == "valid") != (expected_index in candidates_by_index):
            return "valid slots and scored candidates do not match"
        attempts = slot.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            return "a slot has no auditable attempts"
        try:
            _usage(slot.get("usage", {}))
        except RuntimeError as exc:
            return str(exc)
    baseline = candidates_by_index.get(1)
    baseline_id = baseline.get("candidate_id") if baseline else None
    if result.get("baseline_candidate_id") != baseline_id:
        return "Direct Generation does not match C1"
    if candidates:
        selected = max(
            candidates,
            key=lambda row: float(row.get("reward", {}).get("total", 0.0)),
        )
        selected_id = selected.get("candidate_id")
    else:
        selected_id = None
    if result.get("stars_candidate_id") != selected_id:
        return "STARS is not argmax Reward over valid C1-C4 candidates"
    return ""


def _validate_resume_rows(
    existing_results: dict[str, dict[str, Any]],
    expected_sample_keys: list[str],
    protocol_fingerprint: str,
) -> None:
    unexpected = sorted(set(existing_results) - set(expected_sample_keys))
    mismatched = sorted(
        key
        for key, row in existing_results.items()
        if row.get("protocol_fingerprint") != protocol_fingerprint
    )
    if unexpected or mismatched:
        raise RuntimeError(
            "Resume rows do not match the requested protocol: "
            f"unexpected={unexpected[:5]}, mismatched={mismatched[:5]}."
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
    try:
        usage = {
            key: int(raw[key])
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Slot usage must include integer input, output, and total tokens."
        ) from exc
    if any(value < 0 for value in usage.values()):
        raise RuntimeError("Token counts must be non-negative.")
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise RuntimeError("Total tokens must equal input plus output tokens.")
    return usage


def _runner_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
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
