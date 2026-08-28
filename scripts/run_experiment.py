from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.config import ExperimentConfig, FORMAL_REWARD_WEIGHTS
from creative_video_exp.checkpoint_identity import (
    API_MODEL_KIND,
    LOCAL_CHECKPOINT_KIND,
    validate_checkpoint_identity,
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
from creative_video_exp.reporting import print_metrics_summary, write_markdown_report
from creative_video_exp.provenance import build_protocol_manifest, prepare_protocol_manifest
from creative_video_exp.reward import SelfRewardScorer, build_preference_pairs
from creative_video_exp.semantic_points import (
    FORMAL_REFERENCE_PROTOCOL,
    SemanticPointEvaluator,
    formal_reference_contract_issue,
)
from creative_video_exp.text_metrics import evaluate_generation_quality
from creative_video_exp.utils import (
    append_jsonl,
    atomic_write_text,
    ensure_dir,
    iter_jsonl,
    set_seed,
    stable_hash_int,
    write_json,
    write_jsonl,
)
from creative_video_exp.video import SparseFrameBatch, SparseFrameSampler


FORMAL_PROMPT_VERSION = "starsv19_visual_only_contract_guided_v2"
FORMAL_PARSE_RETRY_COUNT = 7
FORMAL_OUTPUT_LANGUAGE = "English"
FORMAL_PRE_SCORE_PROCESSING = (
    "json_envelope_schema_canonicalization_and_timeline_duration_normalization"
)
FORMAL_GENERATION_ENDPOINTS = {
    "llava_video_7b_qwen2": {
        "name": "LLaVA-Video-7B-Qwen2",
        "role": "main_multimodal_generator",
    },
    "internvl25_8b": {
        "name": "InternVL2.5-8B",
        "role": "baseline_multimodal_generator",
    },
    "videollama2_7b_16f": {
        "name": "VideoLLaMA2-7B-16F",
        "role": "baseline_multimodal_generator",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sparse creative video experiment.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse verified results.jsonl rows and process only missing video_ids. "
            "A non-empty formal output directory is otherwise never overwritten."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print one English progress record after this many newly processed samples.",
    )
    return parser.parse_args()


def _validate_formal_config(config: ExperimentConfig) -> None:
    if config.experiment_version != "starsv20":
        raise RuntimeError(
            f"Expected the STARS experiment protocol, got `{config.experiment_version}`."
        )
    if config.generation.prompt_version != FORMAL_PROMPT_VERSION:
        raise RuntimeError(
            "STARS requires generation.prompt_version="
            f"`{FORMAL_PROMPT_VERSION}`."
        )
    if config.generation.parse_retry_count != FORMAL_PARSE_RETRY_COUNT:
        raise RuntimeError(
            "STARS requires generation.parse_retry_count="
            f"{FORMAL_PARSE_RETRY_COUNT}."
        )
    if config.generation.output_language != FORMAL_OUTPUT_LANGUAGE:
        raise RuntimeError(
            "STARS requires generation.output_language="
            f"`{FORMAL_OUTPUT_LANGUAGE}`."
        )
    if config.generation.input_protocol != "visual_only":
        raise RuntimeError(
            "STARS requires generation.input_protocol=`visual_only`."
        )
    if config.models.mode != "server_full_matrix":
        raise RuntimeError("STARS requires the server experiment model suite.")
    if not config.evaluation.semantic_point_coverage_enabled:
        raise RuntimeError("STARS requires Semantic Point Coverage evaluation.")
    if config.long_video.enabled:
        raise RuntimeError("The STARS experiment protocol requires long_video.enabled=false.")
    if config.data.offset != 0 or config.data.limit != 0:
        raise RuntimeError(
            "STARS requires data.offset=0 and data.limit=0."
        )
    if (
        config.data.sample_unit != "unique_video"
    ):
        raise RuntimeError(
            "STARS requires data.sample_unit=`unique_video`."
        )
    if config.generation.num_candidates != 4:
        raise RuntimeError(
            "STARS requires exactly four requested candidate slots."
        )
    if (
        config.sampling.strategy != "binwise"
        or config.sampling.num_bins != 8
        or config.sampling.frames_per_bin != 2
        or config.sampling.max_frames != 16
        or config.sampling.synthetic_size != 224
        or config.sampling.allow_full_decode_fallback
    ):
        raise RuntimeError(
            "STARS requires bin-wise sampling with eight bins, two frames per bin, "
            "a 16-frame budget, and 224-pixel decoded frames."
        )
    if config.generation.candidate_generation_protocol != "independent_single_candidate_calls":
        raise RuntimeError(
            "STARS requires four independent one-candidate generation calls."
        )
    if (
        config.generation.pre_score_processing
        != FORMAL_PRE_SCORE_PROCESSING
    ):
        raise RuntimeError(
            "STARS requires deterministic JSON-envelope, schema, and timeline "
            "duration normalization before reward scoring."
        )
    if (
        config.generation.candidate_slot_failure_policy
        != "retain_invalid_slot_and_continue"
    ):
        raise RuntimeError(
            "STARS requires candidate_slot_failure_policy="
            "`retain_invalid_slot_and_continue`."
        )
    if (
        config.generation.method_failure_aggregation
        != "conditional_and_failure_aware_effective"
    ):
        raise RuntimeError(
            "STARS requires method_failure_aggregation="
            "`conditional_and_failure_aware_effective`."
        )
    generation_contract = {
        "target_duration_sec": 30,
        "segments": 5,
        "summary_position": "late",
        "pace": "medium",
        "information_density": "medium",
        "low_density_chars_per_second": 6.0,
        "medium_density_chars_per_second": 12.0,
        "high_density_chars_per_second": 18.0,
        "density_length_tolerance": 0.5,
        "pace_duration_profile": [],
        "salient_points": [],
        "risk_terms": [
            "absolute",
            "guaranteed",
            "cure",
            "risk-free",
            "lowest price",
            "number one",
            "best",
        ],
    }
    for field, expected in generation_contract.items():
        if getattr(config.generation, field) != expected:
            raise RuntimeError(f"STARS requires generation.{field}={expected!r}.")
    if config.seed != 42:
        raise RuntimeError("STARS requires seed=42.")
    reward_contract = {
        "preference_margin": 0.08,
        "visual_grounding_balance": 0.5,
        "text_anchor_semantic_balance": 0.7,
        "readability_balance_beta": 0.35,
        "readability_sentence_bonus": 0.08,
        "rhythm_evenness_balance": 0.55,
        "numerical_epsilon": 1e-8,
    }
    for component, expected in FORMAL_REWARD_WEIGHTS.items():
        actual = config.reward.component_weights()[component]
        if abs(actual - expected) > 1e-12:
            raise RuntimeError(
                f"STARS requires reward.{component}_weight={expected!r}."
            )
    for field, expected in reward_contract.items():
        if abs(float(getattr(config.reward, field)) - expected) > 1e-12:
            raise RuntimeError(f"STARS requires reward.{field}={expected!r}.")
    if config.evaluation.semantic_point_similarity_threshold != 0.50:
        raise RuntimeError(
            "STARS requires evaluation.semantic_point_similarity_threshold=0.50."
        )
    if config.evaluation.semantic_point_text_fields != [
        "narration",
        "on_screen_text",
        "salient_point",
    ]:
        raise RuntimeError(
            "STARS requires Semantic Point Coverage over narration, on_screen_text, "
            "and salient_point."
        )
    _validate_formal_checkpoint_identities(config)
    if not config.models.allow_heavy_model_load:
        raise RuntimeError("STARS requires the frozen reward encoder to be enabled.")
    generation_endpoint = config.models.get(config.models.active_video_model)
    if generation_endpoint is None:
        raise RuntimeError("The active generation endpoint is missing.")
    generation_identity = FORMAL_GENERATION_ENDPOINTS.get(generation_endpoint.id)
    if (
        generation_identity is None
        or generation_endpoint.name != generation_identity["name"]
        or generation_endpoint.role != generation_identity["role"]
        or generation_endpoint.provider != "openai_compatible"
        or generation_endpoint.adapter != "chat_completions_multimodal"
        or not generation_endpoint.enabled
        or not generation_endpoint.endpoint_url
        or generation_endpoint.checkpoint_identity.get("kind")
        != LOCAL_CHECKPOINT_KIND
    ):
        raise RuntimeError(
            "The active generator must be one of the three local STARS backbones."
        )
    generation_endpoint_contract = {
        "max_frames": 16,
        "max_new_tokens": 900,
        "temperature": 0.3,
        "request_timeout_sec": 300,
        "retry_count": 2,
        "seed": 42,
    }
    for field, expected in generation_endpoint_contract.items():
        if getattr(generation_endpoint, field) != expected:
            raise RuntimeError(
                f"STARS requires the generation endpoint {field}={expected!r}."
            )
    reward_endpoint = config.models.get(config.models.active_reward_vision_model)
    if reward_endpoint is None:
        raise RuntimeError("The active reward endpoint is missing.")
    if (
        reward_endpoint.id != "siglip2_so400m_patch14_384_reward"
        or reward_endpoint.name != "google/siglip2-so400m-patch14-384"
        or reward_endpoint.role != "frozen_vision_text_reward_encoder"
        or reward_endpoint.adapter != "siglip2_frame_encoder"
        or reward_endpoint.provider != "huggingface_local"
        or not reward_endpoint.enabled
        or not reward_endpoint.local_path
        or reward_endpoint.dtype != "bfloat16"
        or reward_endpoint.trust_remote_code
        or reward_endpoint.checkpoint_identity.get("kind")
        != LOCAL_CHECKPOINT_KIND
    ):
        raise RuntimeError("The active reward endpoint does not match the STARS protocol.")


def _strict_video_issue(batch: SparseFrameBatch) -> str:
    if batch.source_kind == "synthetic":
        return "video decoding produced synthetic fallback frames"
    strategy = str(batch.metadata.get("decode_strategy", ""))
    if bool(batch.metadata.get("fallback_used", False)):
        return f"video decoding used fallback strategy `{strategy or 'unknown'}`"
    if strategy not in {"duration_seek_uniform", "preextracted_frames"}:
        return f"video decoding used non-uniform strategy `{strategy or 'unknown'}`"
    return ""


def _validate_formal_checkpoint_identities(config: ExperimentConfig) -> None:


    if config.models.mode != "server_full_matrix":

        return
    active_endpoints = {
        "generation": config.models.active_video_model,
        "reward": config.models.active_reward_vision_model,
    }
    for purpose, endpoint_id in active_endpoints.items():
        if not endpoint_id:
            raise RuntimeError(
                f"Formal {purpose} model endpoint ID is empty."
            )
        endpoint = config.models.get(endpoint_id)
        if endpoint is None or not endpoint.enabled:
            raise RuntimeError(
                f"Formal {purpose} endpoint `{endpoint_id}` is missing or disabled."
            )
        try:
            validate_checkpoint_identity(endpoint.checkpoint_identity)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Formal {purpose} endpoint `{endpoint_id}` has an invalid "
                f"checkpoint_identity: {exc}"
            ) from exc
        expected_kind = (
            API_MODEL_KIND
            if endpoint.provider == "openai"
            else LOCAL_CHECKPOINT_KIND
        )
        if endpoint.provider not in {
            "openai",
            "openai_compatible",
            "huggingface_local",
        }:
            raise RuntimeError(
                f"Formal {purpose} endpoint `{endpoint_id}` uses unsupported provider "
                f"`{endpoint.provider}`."
            )
        identity = endpoint.checkpoint_identity
        if identity.get("kind") != expected_kind:
            raise RuntimeError(
                f"Formal {purpose} endpoint `{endpoint_id}` requires checkpoint "
                f"identity kind `{expected_kind}`."
            )
        if str(identity.get("model_id", "")) != str(endpoint.name):
            raise RuntimeError(
                f"Formal {purpose} endpoint `{endpoint_id}` checkpoint model_id does "
                "not match the requested endpoint name."
            )


def _validate_runtime_reward_checkpoint_identity(
    config: ExperimentConfig,
    encoder_report: dict,
) -> None:


    if config.models.mode != "server_full_matrix":
        return
    endpoint_id = config.models.active_reward_vision_model
    endpoint = config.models.get(endpoint_id)
    if endpoint is None:
        raise RuntimeError(
            f"Runtime reward checkpoint audit cannot resolve endpoint `{endpoint_id}`."
        )
    expected = endpoint.checkpoint_identity
    observed = encoder_report.get("checkpoint_identity")
    if observed != expected:
        raise RuntimeError(
            "Runtime reward checkpoint identity differs from the protocol-bound "
            f"identity for endpoint `{endpoint_id}`."
        )
    if encoder_report.get("checkpoint_identity_sha256") != expected.get(
        "identity_sha256"
    ):
        raise RuntimeError(
            "Runtime reward checkpoint fingerprint is missing or differs from the "
            f"protocol-bound identity for endpoint `{endpoint_id}`."
        )


def _validate_runtime_generation_checkpoint_identity(
    config: ExperimentConfig,
    generation_trace: list[dict],
) -> None:


    if config.models.mode != "server_full_matrix":
        return
    endpoint_id = config.models.active_video_model
    endpoint = config.models.get(endpoint_id)
    if endpoint is None:
        raise RuntimeError(
            f"Runtime generation checkpoint audit cannot resolve endpoint `{endpoint_id}`."
        )
    expected = endpoint.checkpoint_identity
    expected_model_id = str(expected.get("model_id", ""))
    expected_local_fields = {
        "checkpoint_identity_sha256": expected.get("identity_sha256"),
        "checkpoint_model_id": expected.get("model_id"),
        "checkpoint_revision": expected.get("revision"),
    }
    for slot_position, slot in enumerate(generation_trace, start=1):
        if not isinstance(slot, dict):
            raise RuntimeError(
                "Runtime generation identity audit found a non-object slot trace."
            )
        candidate_index = int(slot.get("candidate_index", slot_position))
        attempts = slot.get("attempts", [])
        if not isinstance(attempts, list):
            raise RuntimeError(
                f"Runtime generation identity audit found invalid attempts for C{candidate_index}."
            )
        for attempt_position, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, dict):
                raise RuntimeError(
                    "Runtime generation identity audit found a non-object attempt "
                    f"for C{candidate_index}."
                )
            if attempt.get("status") == "endpoint_error":


                continue
            server_metrics = attempt.get("server_metrics")
            if not isinstance(server_metrics, dict):
                raise RuntimeError(
                    "Runtime generation response lacks model-identity metadata for "
                    f"C{candidate_index} attempt {attempt_position}."
                )
            if server_metrics.get("returned_model_id_conflict") is True:
                raise RuntimeError(
                    "Runtime generation response reported conflicting model "
                    f"identities for C{candidate_index} attempt {attempt_position}."
                )
            returned_model_id = str(server_metrics.get("returned_model_id", ""))
            if returned_model_id != expected_model_id:
                raise RuntimeError(
                    "Runtime generation returned model identity differs from the "
                    f"protocol-bound model for C{candidate_index} attempt "
                    f"{attempt_position}: {returned_model_id!r} != "
                    f"{expected_model_id!r}."
                )
            if endpoint.provider in {"openai_compatible", "huggingface_local"}:
                observed_local_fields = {
                    key: server_metrics.get(key) for key in expected_local_fields
                }
                if observed_local_fields != expected_local_fields:
                    raise RuntimeError(
                        "Runtime local generation checkpoint fingerprint differs "
                        "from the protocol-bound identity for "
                        f"C{candidate_index} attempt {attempt_position}."
                    )


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_file(args.config)
    _validate_formal_config(config)
    config.reward.validate_formal_protocol()
    set_seed(config.seed)

    output_dir = ensure_dir(PROJECT_ROOT / config.output_dir)
    protocol_manifest = build_protocol_manifest(config, PROJECT_ROOT)
    protocol_fingerprint = protocol_manifest["protocol_fingerprint"]
    samples = load_samples(config.data, project_root=PROJECT_ROOT)
    expected_sample_keys = _fixed_manifest_sample_keys(samples)
    if config.evaluation.semantic_point_coverage_enabled:
        construction_protocol = (
            protocol_manifest.get("evaluation_definition", {})
            .get("semantic_point_coverage", {})
            .get("reference_construction_protocol", "")
        )
        if construction_protocol != FORMAL_REFERENCE_PROTOCOL:
            raise RuntimeError(
                "Formal SPC gate failed: expected annotation protocol "
                f"`{FORMAL_REFERENCE_PROTOCOL}`, got "
                f"`{construction_protocol or 'missing'}`."
            )
        invalid_semantic_points = [
            sample.video_id
            for sample in samples
            if not sample.semantic_points
            or sample.semantic_point_source_field != "semantic_points"
        ]
        if invalid_semantic_points:
            preview = ", ".join(invalid_semantic_points[:5])
            raise RuntimeError(
                "Formal SPC gate failed: "
                f"{len(invalid_semantic_points)}/{len(samples)} requested samples "
                "lack a non-empty canonical `semantic_points` annotation. "
                f"Examples: {preview}."
            )
        invalid_reference_contract = [
            (sample.video_id, formal_reference_contract_issue(sample.raw))
            for sample in samples
            if formal_reference_contract_issue(sample.raw)
        ]
        if invalid_reference_contract:
            preview = "; ".join(
                f"{video_id}: {issue}"
                for video_id, issue in invalid_reference_contract[:5]
            )
            raise RuntimeError(
                "Formal SPC reference-contract gate failed: "
                f"{len(invalid_reference_contract)}/{len(samples)} requested "
                f"samples are invalid. Examples: {preview}."
            )
    prepare_protocol_manifest(output_dir, protocol_manifest, resume=args.resume)
    requested_sample_count = len(expected_sample_keys)
    checkpoint_path = output_dir / "results.jsonl"
    failure_path = output_dir / "generation_failures.jsonl"
    existing_results = _load_existing_results(checkpoint_path) if args.resume else {}
    if existing_results:
        unexpected_existing_keys = sorted(
            set(existing_results) - set(expected_sample_keys)
        )
        mismatched_existing_fingerprints = sorted(
            sample_key
            for sample_key, row in existing_results.items()
            if row.get("protocol_fingerprint") != protocol_fingerprint
        )
        if unexpected_existing_keys or mismatched_existing_fingerprints:
            raise RuntimeError(
                "Resume rows do not match the fixed STARS manifest and protocol: "
                f"unexpected_sample_keys={unexpected_existing_keys[:5]}; "
                "fingerprint_mismatches="
                f"{mismatched_existing_fingerprints[:5]}."
            )
    if not args.resume:
        atomic_write_text(checkpoint_path, "")
        atomic_write_text(failure_path, "")
    if existing_results:
        samples = [sample for sample in samples if _sample_key(sample) not in existing_results]

    setup_started = time.perf_counter()
    sampler = SparseFrameSampler(config.sampling)
    encoder_load_started = time.perf_counter()
    encoder, encoder_report = build_frame_encoder(config.models)
    encoder_load_seconds = time.perf_counter() - encoder_load_started
    _validate_runtime_reward_checkpoint_identity(config, encoder_report)
    if encoder_report.get("runtime") != "frozen_siglip2_vision_text_reward":
        raise RuntimeError(
            "Formal reward encoder gate failed: expected "
            "`frozen_siglip2_vision_text_reward`, got "
            f"`{encoder_report.get('runtime', 'unknown')}`. "
            "Do not use StatsFrameEncoder outputs for formal results."
        )
    script_generation_model = build_script_generation_model(config.models)
    runtime_setup_seconds = time.perf_counter() - setup_started
    scorer = SelfRewardScorer(config.reward, config.generation)
    semantic_point_evaluator = SemanticPointEvaluator(encoder, config.evaluation)

    results = []
    preference_pairs = []
    flat_candidates = []
    for result in existing_results.values():
        results.append(result)
        for candidate in result.get("candidates", []):
            flat_candidates.append(candidate)
        preference_pairs.extend(
            build_preference_pairs(
                sample_id=result["video_id"],
                candidate_rows=result.get("candidates", []),
                margin=config.reward.preference_margin,
            )
        )

    newly_attempted = 0
    newly_processed = 0
    failed_in_this_invocation: list[str] = []
    for sample in samples:
        newly_attempted += 1
        sample_started = time.perf_counter()
        sample_key = _sample_key(sample)
        sampling_started = time.perf_counter()
        batch = sampler.sample(
            video_path=sample.video_path,
            video_id=sample.video_id,
            metadata={"input_protocol": "visual_only"},
        )
        sampling_wall_seconds = time.perf_counter() - sampling_started
        sampling_audit_seconds = float(
            batch.metadata.get("sampling_audit_seconds", 0.0)
        )
        sampling_seconds = max(
            0.0,
            sampling_wall_seconds - sampling_audit_seconds,
        )
        strict_video_issue = _strict_video_issue(batch)
        if strict_video_issue:
            raise RuntimeError(
                f"Strict video sampling failed for `{sample.video_id}`: "
                f"{strict_video_issue}. "
                f"Resolved video_path=`{sample.video_path}`. "
                "Check data.video_root and data.video_search_dirs."
            )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        encoding_started = time.perf_counter()
        representation = encoder.encode(batch, context_text="")
        reward_encoding_seconds = time.perf_counter() - encoding_started
        semantic_point_reference = semantic_point_evaluator.prepare(
            sample.semantic_points,
            source_field=sample.semantic_point_source_field,
        )
        generation_started = time.perf_counter()
        candidates = script_generation_model.generate(sample, batch, config.generation)
        generation_seconds = time.perf_counter() - generation_started
        generation_trace = list(
            script_generation_model.model_report.get("last_generation_trace", [])
        )
        if not generation_trace:
            raise RuntimeError(
                "The configured multimodal model produced no auditable candidate-slot "
                "trace. This is an infrastructure/protocol error, not a model-output "
                "failure that may be silently replaced."
            )
        generation_source = "multimodal_model"
        _validate_generation_prompt_protocol(
            script_generation_model.model_report,
            expected_slots=config.generation.num_candidates,
        )
        _validate_runtime_generation_checkpoint_identity(
            config,
            generation_trace,
        )

        if len(candidates) < config.generation.num_candidates:
            _record_generation_failure(
                failure_path=failure_path,
                output_dir=output_dir,
                sample=sample,
                sample_key=sample_key,
                dataset=config.data.name,
                expected_candidates=config.generation.num_candidates,
                candidates=candidates,
                model_report=script_generation_model.model_report,
                protocol_fingerprint=protocol_fingerprint,
            )
            failed_in_this_invocation.append(sample.video_id)
            print(
                "Recorded invalid candidate slot(s) and continued the fixed pool: "
                f"attempted={newly_attempted}/{len(samples)}; "
                f"video_id={sample.video_id}; "
                f"valid_candidates={len(candidates)}/"
                f"{config.generation.num_candidates}.",
                flush=True,
            )
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise RuntimeError(f"Candidate identifiers are not unique for `{sample.video_id}`.")
        for candidate in candidates:
            provenance = candidate.generation_provenance or {}
            if (
                provenance.get("parse_status") != "valid_json_candidate"
                or provenance.get("semantic_repair_applied") is not False
                or provenance.get("wrapped_free_text") is not False
                or provenance.get("surrounding_free_text") is not False
                or provenance.get("response_envelope")
                not in {"raw_json", "single_markdown_json_fence"}
                or provenance.get("timeline_normalization", {}).get("policy")
                != "positive_duration_proportional_v1"
                or provenance.get("timeline_normalization", {}).get("status")
                not in {"applied", "unchanged", "skipped"}
            ):
                raise RuntimeError(
                    f"Candidate `{candidate.candidate_id}` failed the structured JSON-candidate gate."
                )
        candidate_rows = []
        reward_scoring_seconds = 0.0
        reward_scoring_seconds_by_slot = [
            0.0 for _ in range(config.generation.num_candidates)
        ]
        metric_evaluation_seconds = 0.0
        seen_candidate_indices: set[int] = set()
        for fallback_index, candidate in enumerate(candidates, start=1):
            provenance = candidate.generation_provenance or {}
            candidate_index = int(provenance.get("candidate_index", fallback_index))
            if not 1 <= candidate_index <= config.generation.num_candidates:
                raise RuntimeError(
                    f"Candidate `{candidate.candidate_id}` has invalid requested slot "
                    f"index {candidate_index}."
                )
            if candidate_index in seen_candidate_indices:
                raise RuntimeError(
                    f"Requested candidate slot {candidate_index} is duplicated for "
                    f"`{sample.video_id}`."
                )
            seen_candidate_indices.add(candidate_index)
            reward_started = time.perf_counter()
            reward = scorer.score(candidate, representation)
            candidate_reward_seconds = time.perf_counter() - reward_started
            reward_scoring_seconds += candidate_reward_seconds
            reward_scoring_seconds_by_slot[candidate_index - 1] = round(
                candidate_reward_seconds, 6
            )
            row = candidate.as_dict()
            row["candidate_index"] = candidate_index
            row["video_id"] = sample.video_id
            row["reward"] = reward.as_dict()
            metric_started = time.perf_counter()
            row["quality"] = evaluate_generation_quality(
                candidate,
                sample,
                semantic_point_evaluator=semantic_point_evaluator,
                semantic_point_reference=semantic_point_reference,
            )
            metric_evaluation_seconds += time.perf_counter() - metric_started
            row["generation_source"] = generation_source
            candidate_rows.append(row)
            flat_candidates.append(row)
        candidate_rows.sort(key=lambda row: int(row["candidate_index"]))

        slot_outcomes = _build_slot_outcomes(
            generation_trace=generation_trace,
            candidate_rows=candidate_rows,
            expected_slots=config.generation.num_candidates,
        )

        selection_n2_started = time.perf_counter()
        prefix_two = [
            row for row in candidate_rows if int(row["candidate_index"]) <= 2
        ]
        best_of_2 = (
            max(prefix_two, key=lambda row: row["reward"]["total"])
            if prefix_two
            else None
        )
        selection_n2_seconds = time.perf_counter() - selection_n2_started
        selection_n4_started = time.perf_counter()
        best = (
            max(candidate_rows, key=lambda row: row["reward"]["total"])
            if candidate_rows
            else None
        )
        selection_n4_seconds = time.perf_counter() - selection_n4_started
        pairs = build_preference_pairs(
            sample_id=sample_key,
            candidate_rows=candidate_rows,
            margin=config.reward.preference_margin,
        )
        preference_pairs.extend(pairs)

        memory = _reward_runner_memory()
        generation_usage = script_generation_model.model_report.get(
            "last_generation_usage", {}
        )
        slot_generation_seconds = list(
            script_generation_model.model_report.get(
                "last_slot_generation_seconds", []
            )
        )
        slot_generation_usage = list(
            script_generation_model.model_report.get("last_slot_usage", [])
        )
        slot_server_metrics = list(
            script_generation_model.model_report.get(
                "last_slot_server_metrics", []
            )
        )
        active_generation_endpoint = config.models.get(
            config.models.active_video_model
        )
        if (
            active_generation_endpoint is not None
            and active_generation_endpoint.provider == "openai_compatible"
        ):
            reported_server_metrics = [
                item for item in slot_server_metrics if item
            ]
            if not reported_server_metrics or any(
                item.get("server_protocol_version") != "starsv20"
                for item in reported_server_metrics
            ):
                raise RuntimeError(
                    "The local VLM service is not reporting the required protocol version. "
                    "Restart the matching STARS service before continuing."
                )
        if len(slot_generation_seconds) != config.generation.num_candidates:
            slot_generation_seconds = [
                generation_seconds / config.generation.num_candidates
                for _ in range(config.generation.num_candidates)
            ]
        if len(slot_generation_usage) != config.generation.num_candidates:
            slot_generation_usage = [
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                for _ in range(config.generation.num_candidates)
            ]
        prefix_seconds = _prefix_sums(slot_generation_seconds)
        prefix_input_tokens = _prefix_sums(
            [usage.get("input_tokens", 0) for usage in slot_generation_usage]
        )
        prefix_output_tokens = _prefix_sums(
            [usage.get("output_tokens", 0) for usage in slot_generation_usage]
        )
        prefix_total_tokens = _prefix_sums(
            [usage.get("total_tokens", 0) for usage in slot_generation_usage]
        )
        reward_prefix_seconds = _prefix_sums(reward_scoring_seconds_by_slot)
        candidate_semantic_attempt_counts = [
            len(slot.get("attempts", [])) for slot in generation_trace
        ]
        candidate_request_counts = [
            int(
                slot.get(
                    "transport_request_count",
                    len(slot.get("attempts", [])),
                )
            )
            for slot in generation_trace
        ]
        if len(candidate_request_counts) != config.generation.num_candidates:
            candidate_request_counts = [
                0 for _ in range(config.generation.num_candidates)
            ]
        if (
            len(candidate_semantic_attempt_counts)
            != config.generation.num_candidates
        ):
            candidate_semantic_attempt_counts = [
                0 for _ in range(config.generation.num_candidates)
            ]
        prefix_request_counts = _prefix_sums(candidate_request_counts)
        prefix_semantic_attempt_counts = _prefix_sums(
            candidate_semantic_attempt_counts
        )
        baseline = next(
            (
                row
                for row in candidate_rows
                if int(row["candidate_index"]) == 1
            ),
            None,
        )
        total_pipeline_wall_seconds = time.perf_counter() - sample_started
        total_pipeline_seconds = max(
            0.0,
            total_pipeline_wall_seconds - sampling_audit_seconds,
        )
        result_row = {
            "experiment_version": config.experiment_version,
            "protocol_fingerprint": protocol_fingerprint,
            "input_protocol": config.generation.input_protocol,
            "prompt_version": config.generation.prompt_version,
            "prompt_sha256": script_generation_model.model_report.get(
                "last_prompt_sha256", ""
            ),
            "prompt_sha256s": script_generation_model.model_report.get(
                "prompt_sha256s", []
            ),
            "prompt_templates": script_generation_model.model_report.get(
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
            "evaluation_only_references": sample.evaluation_references,
            "source_annotation_row_count": sample.source_annotation_row_count,
            "source_row_fingerprints": sample.source_row_fingerprints,
            "semantic_point_evaluation": semantic_point_reference.audit_dict(),
            "sampling": {
                "source_kind": batch.source_kind,
                "selected_indices": batch.selected_indices,
                "bin_ids": batch.bin_ids,
                "reward_encoder_input_frame_count": len(batch.frames),
                **batch.metadata,
            },
            "representation": {
                "content_tags": representation.content_tags,
                "keyframe_source_indices": representation.keyframe_source_indices,
                "keyframe_candidate_pool_indices": (
                    representation.keyframe_candidate_pool_indices
                ),
                "aggregation_weights": [
                    round(value, 6) for value in representation.aggregation_weights
                ],
                "diagnostics": representation.diagnostics,
            },
            "best_candidate": best,
            "baseline_candidate_id": (
                baseline["candidate_id"] if baseline is not None else None
            ),
            "best_of_1_candidate_id": (
                baseline["candidate_id"] if baseline is not None else None
            ),
            "best_of_2_candidate_id": (
                best_of_2["candidate_id"] if best_of_2 is not None else None
            ),
            "stars_candidate_id": (
                best["candidate_id"] if best is not None else None
            ),
            "method_output_status": {
                "direct_generation_c1": "success" if baseline else "failure",
                "best_of_1": "success" if baseline else "failure",
                "best_of_2": "success" if best_of_2 else "failure",
                "stars_best_of_4": "success" if best else "failure",
            },
            "requested_candidate_slots": config.generation.num_candidates,
            "valid_candidate_slots": len(candidate_rows),
            "failed_candidate_slots": (
                config.generation.num_candidates - len(candidate_rows)
            ),
            "full_candidate_pool": (
                len(candidate_rows) == config.generation.num_candidates
            ),
            "candidate_pool_size": len(candidate_rows),
            "selection_rule": (
                "argmax_full_self_reward_over_valid_candidates_within_requested_prefix"
            ),
            "slot_outcomes": slot_outcomes,
            "candidates": candidate_rows,
            "generation_source": generation_source,
            "efficiency": {
                "sampling_seconds": round(sampling_seconds, 6),
                "sampling_wall_seconds": round(sampling_wall_seconds, 6),
                "sampling_audit_seconds": round(sampling_audit_seconds, 6),
                "reward_encoding_seconds": round(reward_encoding_seconds, 6),
                "generation_seconds": round(generation_seconds, 6),
                "generation_requests": int(sum(candidate_request_counts)),
                "generation_transport_requests": int(
                    sum(candidate_request_counts)
                ),
                "generation_semantic_attempts": int(
                    sum(candidate_semantic_attempt_counts)
                ),
                "generation_input_tokens": int(
                    generation_usage.get("input_tokens", 0)
                ),
                "generation_output_tokens": int(
                    generation_usage.get("output_tokens", 0)
                ),
                "generation_total_tokens": int(
                    generation_usage.get("total_tokens", 0)
                ),
                "generation_seconds_per_requested_slot": round(
                    generation_seconds / config.generation.num_candidates, 6
                ),
                "generation_seconds_per_candidate": round(
                    generation_seconds / config.generation.num_candidates, 6
                ),
                "generation_seconds_per_valid_candidate": round(
                    generation_seconds / max(1, len(candidates)), 6
                ),
                "candidate_generation_seconds_including_retries": [
                    round(float(value), 6) for value in slot_generation_seconds
                ],
                "candidate_generation_usage_including_retries": slot_generation_usage,
                "candidate_server_metrics": slot_server_metrics,
                "slot_generation_seconds_including_retries": [
                    round(float(value), 6) for value in slot_generation_seconds
                ],
                "slot_generation_usage_including_retries": slot_generation_usage,
                "slot_server_metrics": slot_server_metrics,
                "vlm_server_peak_allocated_mib": round(
                    max(
                        [
                            float(item.get("peak_allocated_mib", 0.0))
                            for item in slot_server_metrics
                        ]
                        or [0.0]
                    ),
                    3,
                ),
                "vlm_server_peak_reserved_mib": round(
                    max(
                        [
                            float(item.get("peak_reserved_mib", 0.0))
                            for item in slot_server_metrics
                        ]
                        or [0.0]
                    ),
                    3,
                ),
                "reward_scoring_seconds": round(reward_scoring_seconds, 6),
                "reward_scoring_seconds_per_candidate": round(
                    reward_scoring_seconds / max(1, len(candidates)), 6
                ),
                "metric_evaluation_seconds": round(metric_evaluation_seconds, 6),
                "reward_scoring_seconds_by_slot": reward_scoring_seconds_by_slot,
                "selection_n2_seconds": round(selection_n2_seconds, 9),
                "selection_n4_seconds": round(selection_n4_seconds, 9),
                "online_baseline_seconds": round(
                    sampling_seconds + prefix_seconds[0], 6
                ),
                "online_best_of_2_seconds": round(
                    sampling_seconds
                    + prefix_seconds[1]
                    + reward_encoding_seconds
                    + reward_prefix_seconds[1]
                    + selection_n2_seconds,
                    6,
                ),
                "online_best_of_4_seconds": round(
                    sampling_seconds
                    + prefix_seconds[3]
                    + reward_encoding_seconds
                    + reward_prefix_seconds[3]
                    + selection_n4_seconds,
                    6,
                ),
                "best_of_1_input_tokens": int(prefix_input_tokens[0]),
                "best_of_2_input_tokens": int(prefix_input_tokens[1]),
                "best_of_4_input_tokens": int(prefix_input_tokens[3]),
                "best_of_1_output_tokens": int(prefix_output_tokens[0]),
                "best_of_2_output_tokens": int(prefix_output_tokens[1]),
                "best_of_4_output_tokens": int(prefix_output_tokens[3]),
                "best_of_1_total_tokens": int(prefix_total_tokens[0]),
                "best_of_2_total_tokens": int(prefix_total_tokens[1]),
                "best_of_4_total_tokens": int(prefix_total_tokens[3]),
                "best_of_1_generation_requests": int(prefix_request_counts[0]),
                "best_of_2_generation_requests": int(prefix_request_counts[1]),
                "best_of_4_generation_requests": int(prefix_request_counts[3]),
                "best_of_1_semantic_attempts": int(
                    prefix_semantic_attempt_counts[0]
                ),
                "best_of_2_semantic_attempts": int(
                    prefix_semantic_attempt_counts[1]
                ),
                "best_of_4_semantic_attempts": int(
                    prefix_semantic_attempt_counts[3]
                ),
                "total_pipeline_seconds": round(total_pipeline_seconds, 6),
                "total_pipeline_wall_seconds": round(
                    total_pipeline_wall_seconds,
                    6,
                ),
                "sampled_frames": len(batch.frames),
                **memory,
            },
        }
        results.append(result_row)
        append_jsonl(checkpoint_path, result_row)
        newly_processed += 1
        if args.progress_every > 0 and newly_attempted % args.progress_every == 0:
            print(
                f"Progress: attempted {newly_attempted}/{len(samples)} new samples; "
                f"completed={newly_processed}; failed={len(failed_in_this_invocation)}; "
                f"latest video_id={sample.video_id}; valid_candidates={len(candidate_rows)}.",
                flush=True,
            )

    metrics = summarize_results(results, preference_pairs)
    metrics["model_runtime"] = {
        "video_model": {
            **encoder_report,
            "load_seconds": round(encoder_load_seconds, 6),
        },
        "script_generation_model": script_generation_model.model_report,
        "runner_setup_seconds": round(runtime_setup_seconds, 6),
    }
    metrics["config"] = config.as_dict()
    metrics["protocol"] = protocol_manifest
    metrics["runtime_environment"] = _runtime_environment()
    terminal_failure_audit = _generation_failure_audit(
        failure_path=failure_path,
        results=results,
        requested_sample_count=requested_sample_count,
        expected_sample_keys=expected_sample_keys,
        expected_protocol_fingerprint=protocol_fingerprint,
    )
    provisional_audit = {
        **terminal_failure_audit,
        "terminal_slot_audit_eligible": bool(
            terminal_failure_audit["evidence_eligible"]
        ),
        "failure_aware_analysis_completed": False,
        "failure_aware_analysis_issue": "canonical analysis has not completed",
        "evidence_eligible": False,
    }
    metrics.update(provisional_audit)
    provisional_run_status = {
        "experiment_version": config.experiment_version,
        "protocol_fingerprint": protocol_fingerprint,
        "status": "pending_failure_aware_analysis",
        **provisional_audit,
    }
    write_json(output_dir / "run_status.json", provisional_run_status)

    results = sorted(results, key=lambda row: (str(row.get("video_id", "")), str(row.get("sample_key", ""))))
    write_jsonl(output_dir / "results.jsonl", results)
    write_jsonl(output_dir / "preference_pairs.jsonl", preference_pairs)
    write_json(output_dir / "metrics.json", metrics)
    write_markdown_report(output_dir / "metrics_report.md", metrics)

    candidate_summary: dict = {}
    candidate_selections: list[dict] = []
    failure_aware_analysis_issue = "no requested-sample rows were analyzed"
    if results:
        candidate_summary, candidate_selections = analyze_failure_aware_candidate_pool(
            results=results,
            reward_config=config.reward.__dict__,
            pool_size=4,
            bootstrap_replicates=1000,
            seed=config.seed,
        )
        write_failure_aware_candidate_pool_analysis(
            output_dir / "candidate_pool_analysis",
            candidate_summary,
            candidate_selections,
        )
        metrics["candidate_pool_analysis_path"] = str(
            output_dir
            / "candidate_pool_analysis"
            / "failure_aware_candidate_pool_analysis.json"
        )
        metrics["failure_aware_analysis_protocol"] = candidate_summary.get(
            "analysis_protocol", ""
        )
        metrics["method_output_success_rates"] = {
            method: payload.get("selection_success_rate")
            for method, payload in candidate_summary.get("methods", {}).items()
        }
        metrics["method_conditional_means"] = {
            method: payload.get("conditional_means", {})
            for method, payload in candidate_summary.get("methods", {}).items()
        }
        metrics["method_effective_means"] = {
            method: payload.get("effective_means", {})
            for method, payload in candidate_summary.get("methods", {}).items()
        }
        failure_aware_analysis_issue = _failure_aware_analysis_issue(
            summary=candidate_summary,
            selections=candidate_selections,
            results=results,
            failure_audit=terminal_failure_audit,
            expected_protocol_fingerprint=protocol_fingerprint,
        )
    else:
        metrics["candidate_pool_analysis_path"] = ""
    failure_audit = {
        **terminal_failure_audit,
        "terminal_slot_audit_eligible": bool(
            terminal_failure_audit["evidence_eligible"]
        ),
        "failure_aware_analysis_completed": not failure_aware_analysis_issue,
        "failure_aware_analysis_issue": failure_aware_analysis_issue,
        "evidence_eligible": bool(
            terminal_failure_audit["evidence_eligible"]
            and not failure_aware_analysis_issue
        ),
    }
    metrics.update(failure_audit)
    metrics["verified_invariants"] = {
        "direct_generation_equals_best_of_1_including_failure": bool(
            candidate_summary.get("baseline_best_of_1_invariant", {}).get(
                "identical_failure_states", False
            )
            if results
            else False
        ),
        "nested_failure_aware_best_of_reward_monotonicity": bool(
            candidate_summary.get("nested_reward_monotonicity", {}).get(
                "verified", False
            )
            if results
            else False
        ),
        "all_requested_candidate_slots_accounted": (
            failure_audit["accounted_candidate_slot_count"]
            == failure_audit["requested_candidate_slot_count"]
        ),
        "failed_slots_retained_as_invalid": True,
        "semantic_repair_before_scoring": False,
        "free_text_wrapping": False,
        "deterministic_json_envelope_normalization": True,
        "deterministic_schema_envelope_normalization": True,
        "model_control_metadata_used_for_scoring": False,
        "control_violations_preserved_before_scoring": True,
        "surrounding_free_text_accepted": False,
        "fixed_manifest_fully_accounted": bool(failure_audit["evidence_eligible"]),
        "failed_sample_replacement": False,
    }
    run_status = {
        "experiment_version": config.experiment_version,
        "protocol_fingerprint": protocol_fingerprint,
        "status": "eligible" if failure_audit["evidence_eligible"] else "incomplete",
        **failure_audit,
    }
    write_json(output_dir / "run_status.json", run_status)
    write_json(output_dir / "metrics.json", metrics)
    write_markdown_report(output_dir / "metrics_report.md", metrics)

    print(f"Requested samples: {requested_sample_count}", flush=True)
    print(f"Attempted new samples: {newly_attempted}", flush=True)
    print(f"Completed samples: {len(results)}", flush=True)
    print(
        f"Unresolved generation failures: "
        f"{failure_audit['unresolved_generation_failure_count']}",
        flush=True,
    )
    print(f"Evidence eligible: {failure_audit['evidence_eligible']}", flush=True)
    print(f"Candidates: {len(flat_candidates)}")
    print(f"Preference pairs: {len(preference_pairs)}")
    print(f"Mean best reward: {metrics.get('mean_best_reward', 0.0)}")
    print(f"Control success best: {metrics.get('control_success_rate_best', 0.0)}")
    print(f"Outputs: {output_dir}")
    print_metrics_summary(metrics)


def _load_existing_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    if path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                raise RuntimeError(
                    "Resume results end with an incomplete JSONL row. Refusing to "
                    f"discard or replace checkpoint data in `{path}`."
                )
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Resume results contain invalid JSON at line {row_number} "
                    f"in `{path}`."
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"Resume results row {row_number} is not a JSON object in `{path}`."
                )
            sample_key = row.get("sample_key") or _sample_key_from_row(row)
            if sample_key:
                row["sample_key"] = sample_key
                sample_key = str(sample_key)
                if sample_key in rows:
                    raise RuntimeError(
                        "Resume results contain a duplicate sample_key: "
                        f"`{sample_key}`. Refuse silent last-row-wins replacement."
                    )
                rows[sample_key] = row
    return rows


def _build_slot_outcomes(
    generation_trace: list[dict],
    candidate_rows: list[dict],
    expected_slots: int,
) -> list[dict]:


    if len(generation_trace) != expected_slots:
        raise RuntimeError(
            f"Expected {expected_slots} terminal slot traces, received "
            f"{len(generation_trace)}."
        )
    candidate_by_index = {
        int(candidate["candidate_index"]): candidate for candidate in candidate_rows
    }
    outcomes: list[dict] = []
    for expected_index, source in enumerate(generation_trace, start=1):
        candidate_index = int(source.get("candidate_index", 0))
        if candidate_index != expected_index:
            raise RuntimeError(
                "Candidate slot traces must be ordered exactly C1 through "
                f"C{expected_slots}; received index {candidate_index} at position "
                f"{expected_index}."
            )
        candidate = candidate_by_index.get(candidate_index)
        terminal_status = "valid" if candidate is not None else "failed"
        if bool(source.get("accepted")) != (candidate is not None):
            raise RuntimeError(
                f"Slot C{candidate_index} acceptance does not match its scored "
                "candidate record."
            )
        attempts = list(source.get("attempts", []))
        reason_counts: dict[str, int] = {}
        for attempt in attempts:
            if attempt.get("status") == "accepted":
                continue
            reason = str(
                attempt.get("parse_error")
                or attempt.get("error")
                or attempt.get("status")
                or "unknown_failure"
            )
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        usage = source.get("usage") or {
            key: sum(
                int(attempt.get("usage", {}).get(key, 0))
                for attempt in attempts
            )
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        outcome = dict(source)
        outcome.update(
            {
                "candidate_index": candidate_index,
                "terminal_status": terminal_status,
                "terminal_reason": (
                    str(
                        source.get("terminal_reason")
                        or "accepted_valid_candidate"
                    )
                    if candidate is not None
                    else str(
                        source.get("terminal_reason")
                        or "bounded_attempts_exhausted"
                    )
                ),
                "candidate_id": (
                    candidate["candidate_id"] if candidate is not None else None
                ),
                "request_count": int(
                    source.get("request_count", len(attempts))
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
                "usage": {
                    key: int(usage.get(key, 0))
                    for key in ("input_tokens", "output_tokens", "total_tokens")
                },
                "rejection_reason_counts": dict(sorted(reason_counts.items())),
                "attempts": attempts,
            }
        )
        outcomes.append(outcome)
    if set(candidate_by_index) != {
        int(outcome["candidate_index"])
        for outcome in outcomes
        if outcome["terminal_status"] == "valid"
    }:
        raise RuntimeError("Valid candidate rows and terminal slot outcomes disagree.")
    return outcomes


def _record_generation_failure(
    failure_path: Path,
    output_dir: Path,
    sample,
    sample_key: str,
    dataset: str,
    expected_candidates: int,
    candidates: list,
    model_report: dict,
    protocol_fingerprint: str,
) -> None:
    record = {
        "experiment_version": "starsv20",
        "protocol_fingerprint": protocol_fingerprint,
        "dataset": dataset,
        "sample_key": sample_key,
        "video_id": sample.video_id,
        "video_path": sample.video_path,
        "expected_candidates": expected_candidates,
        "parsed_candidates": len(candidates),
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "candidate_slot_failure_policy": "retain_invalid_slot_and_continue",
        "failed_candidate_slot_indices": [
            int(slot.get("candidate_index", 0))
            for slot in model_report.get("last_generation_trace", [])
            if not slot.get("accepted")
        ],
        "model_report": model_report,
    }
    append_jsonl(failure_path, record)
    write_json(output_dir / "latest_generation_failure.json", record)
    print(
        "Invalid candidate-slot trace recorded; later slots and samples continue: "
        f"video_id={sample.video_id}; parsed={len(candidates)}/{expected_candidates}; "
        f"path={failure_path}.",
        flush=True,
    )


def _validate_generation_prompt_protocol(
    model_report: dict,
    expected_slots: int,
) -> None:


    base_hash = str(model_report.get("base_prompt_sha256", ""))
    retry_hash = str(model_report.get("validation_retry_prompt_sha256", ""))
    trace = list(model_report.get("last_generation_trace", []))
    recorded_hashes = list(model_report.get("prompt_sha256s", []))
    if not base_hash or not retry_hash or base_hash == retry_hash:
        raise RuntimeError("Missing distinct fixed base/retry prompt hashes.")
    if len(trace) != expected_slots:
        raise RuntimeError(
            f"Expected {expected_slots} candidate prompt traces, received {len(trace)}."
        )

    trace_hashes: list[str] = []
    for slot in trace:
        attempts = list(slot.get("attempts", []))
        if not attempts:
            raise RuntimeError("A candidate slot has no generation attempt record.")
        for index, attempt in enumerate(attempts):
            expected_kind = "base_generation" if index == 0 else "validation_retry"
            expected_hash = base_hash if index == 0 else retry_hash
            if attempt.get("prompt_kind") != expected_kind:
                raise RuntimeError(
                    "Candidate retry prompts do not follow the fixed two-template protocol."
                )
            if attempt.get("prompt_sha256") != expected_hash:
                raise RuntimeError(
                    "Candidate retry prompt hash differs from the fixed protocol hash."
                )
            trace_hashes.append(expected_hash)
    if trace_hashes != recorded_hashes:
        raise RuntimeError("Prompt-hash accounting does not match generation attempts.")


def _nonempty_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _failure_aware_analysis_issue(
    summary: dict,
    selections: list[dict],
    results: list[dict],
    failure_audit: dict,
    expected_protocol_fingerprint: str,
) -> str:


    expected = {
        "experiment_version": "starsv20",
        "analysis_protocol": "failure_aware_fixed_requested_slots_v1",
        "protocol_fingerprint": expected_protocol_fingerprint,
        "requested_sample_count": failure_audit["requested_num_samples"],
        "requested_slots_per_sample": 4,
        "requested_candidate_slot_count": failure_audit[
            "requested_candidate_slot_count"
        ],
        "valid_candidate_slot_count": failure_audit[
            "valid_candidate_slot_count"
        ],
        "failed_candidate_slot_count": failure_audit[
            "failed_candidate_slot_count"
        ],
        "full_candidate_pool_sample_count": failure_audit[
            "full_candidate_pool_sample_count"
        ],
        "any_valid_candidate_sample_count": failure_audit[
            "any_valid_candidate_sample_count"
        ],
    }
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            return (
                f"canonical failure-aware analysis {key}={summary.get(key)!r}; "
                f"expected {expected_value!r}"
            )
    result_keys = [str(row.get("sample_key", "")) for row in results]
    selection_keys = [str(row.get("sample_key", "")) for row in selections]
    if (
        len(selection_keys) != len(result_keys)
        or len(set(selection_keys)) != len(selection_keys)
        or set(selection_keys) != set(result_keys)
    ):
        return "canonical failure-aware selections do not match the fixed result rows"
    invariant = summary.get("baseline_best_of_1_invariant", {})
    if not all(
        invariant.get(key)
        for key in (
            "identical_candidate_ids",
            "identical_failure_states",
            "identical_conditional_metrics",
            "identical_effective_metrics",
        )
    ):
        return "Direct Generation and Best-of-1 canonical invariant failed"
    if not summary.get("nested_reward_monotonicity", {}).get("verified"):
        return "canonical nested failure-aware Reward invariant failed"
    return ""


def _generation_failure_audit(
    failure_path: Path,
    results: list[dict],
    requested_sample_count: int,
    expected_sample_keys: list[str],
    expected_protocol_fingerprint: str,
) -> dict:
    records = list(iter_jsonl(failure_path)) if failure_path.exists() else []
    all_failed_video_ids: set[str] = set()
    for record in records:
        video_id = str(record.get("video_id", ""))
        if video_id:
            all_failed_video_ids.add(video_id)

    accounting_issues: list[dict[str, str]] = []
    actual_sample_keys = [str(result.get("sample_key", "")) for result in results]
    sample_key_counts: dict[str, int] = {}
    for sample_key in actual_sample_keys:
        sample_key_counts[sample_key] = sample_key_counts.get(sample_key, 0) + 1
    duplicate_result_sample_keys = sorted(
        sample_key
        for sample_key, count in sample_key_counts.items()
        if sample_key and count > 1
    )
    empty_result_sample_key_count = sum(not sample_key for sample_key in actual_sample_keys)
    expected_sample_key_set = set(expected_sample_keys)
    actual_sample_key_set = {key for key in actual_sample_keys if key}
    missing_manifest_sample_keys = sorted(
        expected_sample_key_set - actual_sample_key_set
    )
    unexpected_result_sample_keys = sorted(
        actual_sample_key_set - expected_sample_key_set
    )
    row_protocol_fingerprint_mismatches = sorted(
        str(result.get("sample_key", ""))
        for result in results
        if result.get("protocol_fingerprint") != expected_protocol_fingerprint
    )
    row_experiment_version_mismatches = sorted(
        str(result.get("sample_key", ""))
        for result in results
        if result.get("experiment_version") != "starsv20"
    )
    expected_manifest_key_contract_valid = (
        len(expected_sample_keys) == requested_sample_count
        and len(expected_sample_key_set) == requested_sample_count
        and all(expected_sample_keys)
    )
    fixed_manifest_row_set_matches = (
        not empty_result_sample_key_count
        and not duplicate_result_sample_keys
        and not missing_manifest_sample_keys
        and not unexpected_result_sample_keys
    )
    row_protocol_integrity_matches = (
        not row_protocol_fingerprint_mismatches
        and not row_experiment_version_mismatches
    )
    manifest_integrity_issue_count = (
        int(not expected_manifest_key_contract_valid)
        + empty_result_sample_key_count
        + len(duplicate_result_sample_keys)
        + len(missing_manifest_sample_keys)
        + len(unexpected_result_sample_keys)
        + len(row_protocol_fingerprint_mismatches)
        + len(row_experiment_version_mismatches)
    )
    slot_outcomes: list[dict] = []
    all_attempts: list[dict] = []
    failed_attempts: list[dict] = []
    samples_with_any_valid = 0
    full_pool_samples = 0
    for result in results:
        sample_key = str(result.get("sample_key", ""))
        outcomes = list(result.get("slot_outcomes", []))
        issue = _slot_accounting_issue(result, expected_slots=4)
        if issue:
            accounting_issues.append({"sample_key": sample_key, "issue": issue})
        slot_outcomes.extend(outcomes)
        valid_count = sum(
            1 for slot in outcomes if slot.get("terminal_status") == "valid"
        )
        if valid_count:
            samples_with_any_valid += 1
        if valid_count == 4:
            full_pool_samples += 1
        for slot in outcomes:
            all_attempts.extend(slot.get("attempts", []))
            if slot.get("terminal_status") != "failed":
                continue
            failed_attempts.extend(slot.get("attempts", []))

    valid_slots = sum(
        1 for slot in slot_outcomes if slot.get("terminal_status") == "valid"
    )
    failed_slots = sum(
        1 for slot in slot_outcomes if slot.get("terminal_status") == "failed"
    )
    accounted_slots = valid_slots + failed_slots
    requested_slots = requested_sample_count * 4

    usage_keys = ("input_tokens", "output_tokens", "total_tokens")
    failure_usage = {
        key: sum(
            int(attempt.get("usage", {}).get(key, 0))
            for attempt in failed_attempts
        )
        for key in usage_keys
    }
    completed = len(results)
    failure_record_count = len(records)
    infrastructure_issues = _infrastructure_attempt_issues(results)
    infrastructure_issue_sample_keys = sorted(
        {
            str(issue.get("sample_key", ""))
            for issue in infrastructure_issues
            if issue.get("sample_key")
        }
    )
    semantic_endpoint_error_attempt_count = sum(
        1 for attempt in all_attempts if attempt.get("status") == "endpoint_error"
    )
    transport_error_attempt_count = sum(
        1
        for attempt in all_attempts
        for transport in attempt.get("transport_attempts", [])
        if transport.get("status") == "error"
    )
    evidence_eligible = (
        completed == requested_sample_count
        and accounted_slots == requested_slots
        and not accounting_issues
        and not infrastructure_issues
        and expected_manifest_key_contract_valid
        and fixed_manifest_row_set_matches
        and row_protocol_integrity_matches
    )
    unaccounted_sample_count = max(0, requested_sample_count - completed)
    unresolved_count = unaccounted_sample_count + len(accounting_issues)
    return {
        "requested_num_samples": requested_sample_count,
        "completed_num_samples": completed,
        "fully_accounted_sample_count": (
            completed - len(accounting_issues)
        ),
        "slot_accounting_completion_rate": round(
            accounted_slots / max(1, requested_slots), 6
        ),
        "requested_candidate_slot_count": requested_slots,
        "accounted_candidate_slot_count": accounted_slots,
        "valid_candidate_slot_count": valid_slots,
        "failed_candidate_slot_count": failed_slots,
        "valid_candidate_slot_rate": round(
            valid_slots / max(1, requested_slots), 6
        ),
        "candidate_slot_success_rate": round(
            valid_slots / max(1, requested_slots), 6
        ),
        "full_candidate_pool_count": full_pool_samples,
        "full_candidate_pool_rate": round(
            full_pool_samples / max(1, requested_sample_count), 6
        ),
        "full_candidate_pool_sample_count": full_pool_samples,
        "full_candidate_pool_sample_rate": round(
            full_pool_samples / max(1, requested_sample_count), 6
        ),
        "samples_with_any_valid_candidate": samples_with_any_valid,
        "samples_with_any_valid_candidate_rate": round(
            samples_with_any_valid / max(1, requested_sample_count), 6
        ),
        "any_valid_candidate_sample_count": samples_with_any_valid,
        "any_valid_candidate_sample_rate": round(
            samples_with_any_valid / max(1, requested_sample_count), 6
        ),
        "any_valid_candidate_rate": round(
            samples_with_any_valid / max(1, requested_sample_count), 6
        ),
        "generation_success_rate": round(
            samples_with_any_valid / max(1, requested_sample_count), 6
        ),
        "generation_failure_records": failure_record_count,
        "unresolved_generation_failure_count": unresolved_count,
        "generation_failure_video_ids": sorted(all_failed_video_ids),
        "unresolved_generation_failure_video_ids": [
            str(item.get("sample_key", "")) for item in accounting_issues
        ],
        "slot_accounting_issues": accounting_issues,
        "expected_manifest_sample_key_count": len(expected_sample_keys),
        "unique_result_sample_key_count": len(actual_sample_key_set),
        "expected_manifest_key_contract_valid": expected_manifest_key_contract_valid,
        "fixed_manifest_row_set_matches": fixed_manifest_row_set_matches,
        "row_protocol_integrity_matches": row_protocol_integrity_matches,
        "manifest_integrity_issue_count": manifest_integrity_issue_count,
        "duplicate_result_sample_keys": duplicate_result_sample_keys,
        "empty_result_sample_key_count": empty_result_sample_key_count,
        "missing_manifest_sample_keys": missing_manifest_sample_keys,
        "unexpected_result_sample_keys": unexpected_result_sample_keys,
        "row_protocol_fingerprint_mismatches": row_protocol_fingerprint_mismatches,
        "row_experiment_version_mismatches": row_experiment_version_mismatches,
        "failed_generation_attempt_records": len(failed_attempts),
        "failed_generation_parse_rejections": sum(
            1
            for attempt in failed_attempts
            if attempt.get("status") == "parse_rejected"
        ),
        "failed_generation_endpoint_errors": sum(
            1
            for attempt in failed_attempts
            if attempt.get("status") == "endpoint_error"
        ),
        "semantic_endpoint_error_attempt_count": (
            semantic_endpoint_error_attempt_count
        ),
        "transport_error_attempt_count": transport_error_attempt_count,
        "infrastructure_issue_count": len(infrastructure_issues),
        "infrastructure_issue_sample_count": len(
            infrastructure_issue_sample_keys
        ),
        "infrastructure_issue_sample_keys": infrastructure_issue_sample_keys,
        "infrastructure_issue_seconds": round(
            sum(
                float(issue.get("request_seconds", 0.0))
                for issue in infrastructure_issues
            ),
            6,
        ),
        "infrastructure_issues": infrastructure_issues,
        "formal_infrastructure_valid": not infrastructure_issues,
        "failed_generation_repetition_early_stops": sum(
            1
            for attempt in failed_attempts
            if attempt.get("server_metrics", {}).get(
                "degenerate_repetition_early_stop"
            )
            is True
        ),
        "failed_generation_seconds": round(
            sum(
                float(attempt.get("request_seconds", 0.0))
                for attempt in failed_attempts
            ),
            6,
        ),
        "failed_generation_input_tokens": failure_usage["input_tokens"],
        "failed_generation_output_tokens": failure_usage["output_tokens"],
        "failed_generation_total_tokens": failure_usage["total_tokens"],
        "evidence_eligible": evidence_eligible,
    }


def _infrastructure_attempt_issues(results: list[dict]) -> list[dict]:


    issues: list[dict] = []
    for result in results:
        sample_key = str(result.get("sample_key", ""))
        video_id = str(result.get("video_id", ""))
        for slot in result.get("slot_outcomes", []):
            candidate_index = int(slot.get("candidate_index", 0))
            for attempt_position, attempt in enumerate(
                slot.get("attempts", []), start=1
            ):
                semantic_attempt = int(
                    attempt.get("attempt", attempt_position)
                )
                nested_errors = [
                    transport
                    for transport in attempt.get("transport_attempts", [])
                    if transport.get("status") == "error"
                ]
                for transport in nested_errors:
                    issues.append(
                        {
                            "sample_key": sample_key,
                            "video_id": video_id,
                            "candidate_index": candidate_index,
                            "semantic_attempt": semantic_attempt,
                            "transport_attempt": int(
                                transport.get("transport_attempt", 0)
                            ),
                            "source": "nested_transport_attempt",
                            "status": "error",
                            "error_type": str(
                                transport.get("error_type", "")
                            ),
                            "error": str(transport.get("error", "")),
                            "request_seconds": round(
                                float(transport.get("request_seconds", 0.0)),
                                6,
                            ),
                        }
                    )
                if attempt.get("status") == "endpoint_error" and not nested_errors:
                    issues.append(
                        {
                            "sample_key": sample_key,
                            "video_id": video_id,
                            "candidate_index": candidate_index,
                            "semantic_attempt": semantic_attempt,
                            "transport_attempt": None,
                            "source": "semantic_endpoint_error",
                            "status": "endpoint_error",
                            "error_type": "",
                            "error": str(attempt.get("error", "")),
                            "request_seconds": round(
                                float(attempt.get("request_seconds", 0.0)),
                                6,
                            ),
                        }
                    )
    return issues


def _slot_accounting_issue(result: dict, expected_slots: int) -> str:


    try:
        declared_requested = int(result.get("requested_candidate_slots", -1))
        declared_valid = int(result.get("valid_candidate_slots", -1))
        declared_failed = int(result.get("failed_candidate_slots", -1))
    except (TypeError, ValueError):
        return "requested/valid/failed candidate-slot declarations must be integers"
    if declared_requested != expected_slots:
        return (
            "requested_candidate_slots does not match the fixed pool: "
            f"{declared_requested} != {expected_slots}"
        )

    raw_outcomes = result.get("slot_outcomes", [])
    if not isinstance(raw_outcomes, list):
        return "slot_outcomes is not a list"
    outcomes = list(raw_outcomes)
    if len(outcomes) != expected_slots:
        return f"expected {expected_slots} slot outcomes, found {len(outcomes)}"
    try:
        indices = [int(slot.get("candidate_index", 0)) for slot in outcomes]
    except (AttributeError, TypeError, ValueError):
        return "candidate slot indices are missing or non-integer"
    if indices != list(range(1, expected_slots + 1)):
        return f"slot indices are not exactly 1..{expected_slots}: {indices}"
    statuses = [str(slot.get("terminal_status", "")) for slot in outcomes]
    if any(status not in {"valid", "failed"} for status in statuses):
        return f"invalid terminal status sequence: {statuses}"

    raw_candidates = result.get("candidates", [])
    if not isinstance(raw_candidates, list):
        return "candidates is not a list"
    candidates_by_index: dict[int, dict] = {}
    candidate_ids: set[str] = set()
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            return "a scored candidate row is not an object"
        try:
            candidate_index = int(candidate.get("candidate_index", 0))
        except (TypeError, ValueError):
            return "a scored candidate has a non-integer candidate_index"
        candidate_id = str(candidate.get("candidate_id") or "")
        if not 1 <= candidate_index <= expected_slots:
            return f"scored candidate index {candidate_index} lies outside the fixed pool"
        if candidate_index in candidates_by_index:
            return f"scored candidate index C{candidate_index} is duplicated"
        if not candidate_id or candidate_id in candidate_ids:
            return "scored candidate IDs must be non-empty and unique"
        candidates_by_index[candidate_index] = candidate
        candidate_ids.add(candidate_id)

    valid_indices = {
        int(slot["candidate_index"])
        for slot in outcomes
        if slot.get("terminal_status") == "valid"
    }
    candidate_indices = set(candidates_by_index)
    if valid_indices != candidate_indices:
        return (
            "valid slot indices do not match scored candidate indices: "
            f"{sorted(valid_indices)} != {sorted(candidate_indices)}"
        )
    if declared_valid != len(valid_indices):
        return (
            "valid_candidate_slots does not match terminal outcomes: "
            f"{declared_valid} != {len(valid_indices)}"
        )
    if declared_failed != expected_slots - len(valid_indices):
        return (
            "failed_candidate_slots does not match terminal outcomes: "
            f"{declared_failed} != {expected_slots - len(valid_indices)}"
        )

    max_semantic_attempts = FORMAL_PARSE_RETRY_COUNT + 1
    usage_keys = ("input_tokens", "output_tokens", "total_tokens")
    for slot in outcomes:
        candidate_index = int(slot["candidate_index"])
        status = str(slot.get("terminal_status", ""))
        is_valid = status == "valid"
        terminal_reason = str(slot.get("terminal_reason") or "").strip()
        if not terminal_reason:
            return f"slot C{candidate_index} is missing terminal_reason"

        raw_candidate_id = slot.get("candidate_id")
        candidate_id = None if raw_candidate_id is None else str(raw_candidate_id)
        candidate = candidates_by_index.get(candidate_index)
        if is_valid:
            expected_candidate_id = str(candidate.get("candidate_id")) if candidate else ""
            if not candidate_id or candidate_id != expected_candidate_id:
                return (
                    f"slot C{candidate_index} candidate_id does not join exactly to "
                    "its scored candidate"
                )
            if terminal_reason != "accepted_valid_candidate":
                return f"valid slot C{candidate_index} has a failure terminal_reason"
        elif candidate_id is not None or candidate is not None:
            return f"failed slot C{candidate_index} references a scored candidate"

        attempts = slot.get("attempts", [])
        if not isinstance(attempts, list):
            return f"slot C{candidate_index} attempts is not a list"
        try:
            request_count = int(slot.get("request_count", -1))
        except (TypeError, ValueError):
            return f"slot C{candidate_index} request_count is not an integer"
        if not attempts:
            return f"slot C{candidate_index} has no auditable semantic attempts"
        if request_count != len(attempts):
            return (
                f"slot C{candidate_index} request_count does not match semantic "
                "attempt records"
            )
        if is_valid and not 1 <= request_count <= max_semantic_attempts:
            return f"valid slot C{candidate_index} has an invalid semantic-attempt count"
        if not is_valid and request_count != max_semantic_attempts:
            return (
                f"failed slot C{candidate_index} did not exhaust the fixed "
                f"{max_semantic_attempts}-attempt bound"
            )

        semantic_usage = {key: 0 for key in usage_keys}
        semantic_seconds = 0.0
        accepted_positions: list[int] = []
        transport_count = 0
        transport_seconds = 0.0
        transport_backoff_seconds = 0.0
        for position, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, dict):
                return f"slot C{candidate_index} attempt {position} is not an object"
            try:
                recorded_attempt = int(attempt.get("attempt", -1))
            except (TypeError, ValueError):
                recorded_attempt = -1
            if recorded_attempt != position:
                return (
                    f"slot C{candidate_index} semantic attempt indices are not "
                    f"exactly 1..{request_count}"
                )
            attempt_status = str(attempt.get("status", ""))
            if attempt_status not in {"accepted", "parse_rejected", "endpoint_error"}:
                return (
                    f"slot C{candidate_index} attempt {position} has invalid status "
                    f"`{attempt_status}`"
                )
            if attempt_status == "accepted":
                accepted_positions.append(position)
            try:
                attempt_seconds = float(attempt.get("request_seconds", -1.0))
            except (TypeError, ValueError):
                attempt_seconds = -1.0
            if not _is_finite_nonnegative(attempt_seconds):
                return (
                    f"slot C{candidate_index} attempt {position} lacks finite, "
                    "non-negative request_seconds"
                )
            semantic_seconds += attempt_seconds

            attempt_usage = attempt.get("usage", {})
            if attempt_status == "endpoint_error" and not attempt_usage:
                attempt_usage = {key: 0 for key in usage_keys}
            usage_issue, clean_usage = _usage_contract_issue(attempt_usage)
            if usage_issue:
                return (
                    f"slot C{candidate_index} attempt {position} usage: "
                    f"{usage_issue}"
                )
            for key in usage_keys:
                semantic_usage[key] += clean_usage[key]

            raw_transports = attempt.get("transport_attempts")
            if not isinstance(raw_transports, list) or not raw_transports:
                return (
                    f"slot C{candidate_index} attempt {position} has no auditable "
                    "transport_attempts"
                )
            try:
                declared_transport_count = int(
                    attempt.get("transport_request_count", -1)
                )
            except (TypeError, ValueError):
                declared_transport_count = -1
            if declared_transport_count != len(raw_transports):
                return (
                    f"slot C{candidate_index} attempt {position} transport request "
                    "count does not match its trace"
                )
            for transport_position, transport in enumerate(raw_transports, start=1):
                transport_issue = _transport_attempt_contract_issue(
                    transport,
                    expected_index=transport_position,
                )
                if transport_issue:
                    return (
                        f"slot C{candidate_index} attempt {position} transport "
                        f"{transport_position}: {transport_issue}"
                    )
                transport_seconds += float(transport["request_seconds"])
                transport_backoff_seconds += float(
                    transport.get("backoff_seconds_after", 0.0)
                )
            transport_count += len(raw_transports)

        if is_valid:
            if accepted_positions != [request_count]:
                return (
                    f"valid slot C{candidate_index} must end with exactly one accepted "
                    "semantic attempt"
                )
        elif accepted_positions:
            return f"failed slot C{candidate_index} contains an accepted attempt"

        try:
            slot_seconds = float(slot.get("request_seconds", -1.0))
        except (TypeError, ValueError):
            slot_seconds = -1.0
        if (
            not _is_finite_nonnegative(slot_seconds)
            or abs(slot_seconds - semantic_seconds) > 1e-4
        ):
            return f"slot C{candidate_index} request_seconds does not match attempts"
        slot_usage_issue, slot_usage = _usage_contract_issue(slot.get("usage", {}))
        if slot_usage_issue or slot_usage != semantic_usage:
            return f"slot C{candidate_index} usage does not match attempts"
        try:
            declared_transport_count = int(
                slot.get("transport_request_count", -1)
            )
            declared_transport_seconds = float(
                slot.get("transport_request_seconds", -1.0)
            )
            declared_transport_backoff = float(
                slot.get("transport_backoff_seconds", -1.0)
            )
        except (TypeError, ValueError):
            return f"slot C{candidate_index} transport aggregates are non-numeric"
        if declared_transport_count != transport_count:
            return f"slot C{candidate_index} transport_request_count is inconsistent"
        if (
            not _is_finite_nonnegative(declared_transport_seconds)
            or abs(declared_transport_seconds - transport_seconds) > 1e-4
        ):
            return f"slot C{candidate_index} transport_request_seconds is inconsistent"
        if (
            not _is_finite_nonnegative(declared_transport_backoff)
            or abs(declared_transport_backoff - transport_backoff_seconds) > 1e-4
        ):
            return f"slot C{candidate_index} transport_backoff_seconds is inconsistent"
    return ""


def _usage_contract_issue(raw_usage: object) -> tuple[str, dict[str, int]]:
    usage_keys = ("input_tokens", "output_tokens", "total_tokens")
    if not isinstance(raw_usage, dict):
        return "usage is not an object", {key: 0 for key in usage_keys}
    try:
        usage = {key: int(raw_usage[key]) for key in usage_keys}
    except (KeyError, TypeError, ValueError):
        return "all input/output/total token counts must be integers", {
            key: 0 for key in usage_keys
        }
    if any(value < 0 for value in usage.values()):
        return "token counts must be non-negative", usage
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        return "total_tokens must equal input_tokens + output_tokens", usage
    return "", usage


def _transport_attempt_contract_issue(
    transport: object,
    expected_index: int,
) -> str:
    if not isinstance(transport, dict):
        return "record is not an object"
    try:
        observed_index = int(transport.get("transport_attempt", -1))
    except (TypeError, ValueError):
        observed_index = -1
    if observed_index != expected_index:
        return "transport_attempt index is not sequential"
    if transport.get("status") not in {"success", "error"}:
        return "status must be success or error"
    try:
        request_seconds = float(transport.get("request_seconds", -1.0))
        backoff_seconds = float(transport.get("backoff_seconds_after", -1.0))
    except (TypeError, ValueError):
        return "latency fields must be numeric"
    if not _is_finite_nonnegative(request_seconds):
        return "request_seconds must be finite and non-negative"
    if not _is_finite_nonnegative(backoff_seconds):
        return "backoff_seconds_after must be finite and non-negative"
    usage_issue, usage = _usage_contract_issue(transport.get("usage", {}))
    if usage_issue:
        return f"usage {usage_issue}"
    if transport.get("status") == "error":
        if not str(transport.get("error") or "").strip():
            return "error transport lacks an error message"
        if any(usage.values()):
            return "error transport must use zero token counts when no response exists"
    return ""


def _is_finite_nonnegative(value: float) -> bool:
    return value >= 0.0 and value != float("inf") and value == value


def _reward_runner_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "reward_runner_peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / 1024**2, 3
        ),
        "reward_runner_peak_reserved_mib": round(
            torch.cuda.max_memory_reserved() / 1024**2, 3
        ),
    }


def _prefix_sums(values) -> list[float]:
    total = 0.0
    output: list[float] = []
    for value in values:
        total += float(value)
        output.append(total)
    return output


def _runtime_environment() -> dict:
    packages = {}
    for name in ["numpy", "torch", "torchvision", "transformers", "Pillow"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    gpu = ""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_cuda_version": torch.version.cuda or "none",
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": gpu,
        "packages": packages,
        "token_accounting_note": (
            "Local OpenAI-compatible servers report tokenizer-based text token counts; "
            "visual input cost is reported separately as the sampled-frame count."
        ),
    }


def _sample_key(sample) -> str:
    payload = {
        "video_id": sample.video_id,
        "video_path": sample.video_path,
        "source_row_fingerprints": sample.source_row_fingerprints,
    }
    return _sample_key_payload(payload)


def _fixed_manifest_sample_keys(samples) -> list[str]:
    keys = [_sample_key(sample) for sample in samples]
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if not keys or not all(keys) or duplicates:
        raise RuntimeError(
            "The requested fixed manifest must contain non-empty, unique sample keys; "
            f"duplicates={duplicates[:5]}."
        )
    return keys


def _sample_key_from_row(row: dict) -> str:
    payload = {
        "video_id": row.get("video_id", ""),
        "video_path": row.get("video_path", ""),
        "source_row_fingerprints": row.get("source_row_fingerprints", []),
    }
    return _sample_key_payload(payload)


def _sample_key_payload(payload: dict) -> str:
    video_id = str(payload.get("video_id", ""))
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"{video_id}::{stable_hash_int(text, 10**12):012d}"


if __name__ == "__main__":
    main()
