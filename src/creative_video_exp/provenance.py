from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint_identity import (
    LOCAL_CHECKPOINT_KIND,
    checkpoint_identity_summary,
    load_checkpoint_manifest,
    validate_checkpoint_identity,
)
from .config import ExperimentConfig
from .semantic_points import FORMAL_REFERENCE_PROTOCOL
from .utils import resolve_path, sha256_file, sha256_json, write_json


SOURCE_FILES = [
    "scripts/build_checkpoint_manifest.py",
    "scripts/build_semantic_point_manifests.py",
    "scripts/run_experiment.py",
    "scripts/run_server_matrix.py",
    "scripts/serve_internvl_openai.py",
    "scripts/serve_llava_video_openai.py",
    "scripts/serve_videollama2_openai.py",
    "src/creative_video_exp/__init__.py",
    "src/creative_video_exp/checkpoint_identity.py",
    "src/creative_video_exp/config.py",
    "src/creative_video_exp/data.py",
    "src/creative_video_exp/failure_aware_postprocessing.py",
    "src/creative_video_exp/generation.py",
    "src/creative_video_exp/metrics.py",
    "src/creative_video_exp/modeling.py",
    "src/creative_video_exp/provenance.py",
    "src/creative_video_exp/reporting.py",
    "src/creative_video_exp/representations.py",
    "src/creative_video_exp/reward.py",
    "src/creative_video_exp/semantic_points.py",
    "src/creative_video_exp/text_metrics.py",
    "src/creative_video_exp/utils.py",
    "src/creative_video_exp/video.py",
]


def build_protocol_manifest(
    config: ExperimentConfig,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if config.models.mode == "server_full_matrix":
        active_ids = {
            config.models.active_video_model,
            config.models.active_reward_vision_model,
        }
        if "" in active_ids:
            raise RuntimeError(
                "STARS requires active generator and reward model ids."
            )
        for endpoint_id in sorted(active_ids):
            endpoint = config.models.get(endpoint_id)
            if endpoint is None or not endpoint.enabled:
                raise RuntimeError(
                    f"Formal active model endpoint is missing or disabled: {endpoint_id!r}."
                )
            validate_checkpoint_identity(endpoint.checkpoint_identity)
            if endpoint.checkpoint_identity.get("model_id") != endpoint.name:
                raise RuntimeError(
                    "Checkpoint identity model_id must equal the configured endpoint name: "
                    f"{endpoint.checkpoint_identity.get('model_id')!r} != {endpoint.name!r}."
                )
            expected_kind = LOCAL_CHECKPOINT_KIND
            if endpoint.checkpoint_identity.get("kind") != expected_kind:
                raise RuntimeError(
                    f"Endpoint {endpoint_id!r} requires checkpoint kind {expected_kind!r}."
                )
    annotation_path = resolve_path(config.data.annotation_path, root)
    annotation_sha256 = sha256_file(annotation_path) if annotation_path.exists() else ""
    annotation_reference_protocol: dict[str, Any] = {}
    if annotation_path.exists() and annotation_path.suffix.lower() == ".json":
        try:
            annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            annotation_payload = {}
        if isinstance(annotation_payload, dict):
            value = annotation_payload.get("semantic_point_reference_protocol", {})
            if isinstance(value, dict):
                annotation_reference_protocol = value
    endpoints = []
    runtime_endpoints = []
    for endpoint in config.models.endpoints:
        if not endpoint.enabled:
            continue
        checkpoint_manifest: dict[str, Any] = {}
        if endpoint.checkpoint_identity.get("kind") == LOCAL_CHECKPOINT_KIND:
            if not endpoint.checkpoint_manifest_path:
                raise RuntimeError(
                    f"Local endpoint {endpoint.id!r} lacks checkpoint_manifest_path."
                )
            checkpoint_manifest = load_checkpoint_manifest(
                endpoint.checkpoint_manifest_path
            )
            if checkpoint_identity_summary(checkpoint_manifest) != endpoint.checkpoint_identity:
                raise RuntimeError(
                    f"Local endpoint {endpoint.id!r} identity differs from its sidecar."
                )
        endpoints.append({
            "id": endpoint.id,
            "name": endpoint.name,
            "role": endpoint.role,
            "provider": endpoint.provider,
            "adapter": endpoint.adapter,
            "max_frames": endpoint.max_frames,
            "max_new_tokens": endpoint.max_new_tokens,
            "temperature": endpoint.temperature,
            "request_timeout_sec": endpoint.request_timeout_sec,
            "retry_count": endpoint.retry_count,
            "seed": endpoint.seed,
            "device_map": endpoint.device_map,
            "dtype": endpoint.dtype,
            "quantization": endpoint.quantization,
            "trust_remote_code": endpoint.trust_remote_code,
            "checkpoint_identity": endpoint.checkpoint_identity,
            "checkpoint_manifest": checkpoint_manifest,
            "runtime_identity": endpoint.runtime_identity,
        })
        runtime_endpoints.append({
            "id": endpoint.id,
            "local_path": endpoint.local_path,
            "checkpoint_manifest_path": endpoint.checkpoint_manifest_path,
            "endpoint_url": endpoint.endpoint_url,
        })
    actual_source_files = sorted(
        path.relative_to(root).as_posix()
        for directory in (root / "scripts", root / "src")
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    if sorted(SOURCE_FILES) != actual_source_files:
        raise RuntimeError("SOURCE_FILES does not match the retained Python source tree.")
    source_hashes = {
        relative_path: sha256_file(root / relative_path)
        for relative_path in SOURCE_FILES
    }

    scientific_payload = {
        "experiment_version": config.experiment_version,
        "seed": config.seed,
        "data": {
            "name": config.data.name,
            "annotation_sha256": annotation_sha256,
            "semantic_point_reference_protocol": annotation_reference_protocol,
        },
        "sampling": config.sampling.__dict__,
        "generation": config.generation.__dict__,
        "reward": config.reward.__dict__,
        "evaluation": config.evaluation.__dict__,
        "models": {
            "mode": config.models.mode,
            "active_video_model": config.models.active_video_model,
            "active_reward_vision_model": config.models.active_reward_vision_model,
            "endpoints": endpoints,
        },
        "source_hashes": source_hashes,
    }
    configured_weights = {
        key.removesuffix("_weight"): float(value)
        for key, value in config.reward.__dict__.items()
        if key.endswith("_weight")
    }
    weight_sum = sum(value for value in configured_weights.values() if value > 0.0)
    if set(configured_weights) != {"alignment", "readability", "rhythm", "control", "risk"}:
        raise RuntimeError(
            "STARS requires exactly five reward components."
        )
    if abs(weight_sum - 1.0) > 1e-12:
        raise RuntimeError("STARS reward weights must sum to 1.0.")
    return {
        "protocol_fingerprint": sha256_json(scientific_payload),
        "scientific_payload": scientific_payload,
        "annotation_path": str(annotation_path),
        "runtime_locations": {
            "annotation_path": str(annotation_path),
            "endpoints": runtime_endpoints,
        },
        "input_fields": {
            "generation_and_reward": ["sampled_video_frames", "generic_script_contract"],
            "evaluation_only": [
                "question",
                "answer",
                "caption",
                "subtitle",
                "transcript",
                "reference_description",
                "semantic_points",
            ],
        },
        "candidate_pool_protocol": {
            "pool_size": config.generation.num_candidates,
            "generation": config.generation.candidate_generation_protocol,
            "initial_prompt_templates": 1,
            "validation_retry_prompt_templates": 1,
            "retry_prompt_policy": (
                "attempt one uses the shared base prompt; later attempts use one "
                "fixed validation-retry suffix without rejected-response or error "
                "content; the suffix forbids quoting, transcribing, transliterating, "
                "or translating visible text into semantic fields"
            ),
            "parse_retry_count": config.generation.parse_retry_count,
            "max_validation_attempts_per_candidate": (
                config.generation.parse_retry_count + 1
            ),
            "prompt_protocol": {
                "prompt_version": config.generation.prompt_version,
                "input_protocol": config.generation.input_protocol,
                "validation": "fixed deterministic acceptance validation",
            },
            "output_language_contract": (
                "model-authored narration and salient_point must use Latin-script "
                "English; common Unicode punctuation, symbols, units, and Latin-script "
                "names are allowed; on_screen_text may preserve source-visible Unicode "
                "overlays; variant is runner-owned non-semantic metadata"
            ),
            "variant_policy": {
                "owner": "runner",
                "fixed_value": "visual_story",
                "used_by_reward_or_evaluation": False,
                "raw_model_value_preserved_in_attempt_trace": True,
                "canonicalization_record": "fixed_runner_owned_variant",
            },
            "embedded_frame_text_policy": (
                "do not copy multilingual text into narration or salient_point; a brief "
                "clearly visible title, label, or proper name may be preserved verbatim "
                "in on_screen_text and is separately audited"
            ),
            "strict_json_required": True,
            "accepted_response_envelopes": [
                "raw_json",
                "single_markdown_json_fence",
            ],
            "json_envelope_normalization": (
                "remove one exact whole-response markdown json fence only"
            ),
            "schema_envelope_normalization": (
                "when top-level timeline is absent but controls.timeline is a JSON "
                "array, lift that array without changing item order, count, fields, "
                "timestamps, or text; record every occurrence"
            ),
            "nonsemantic_schema_canonicalization": (
                "candidate_id and variant are runner-owned protocol metadata; raw model "
                "responses remain in the attempt trace, and script text, timestamps, "
                "segment count, and control tags are never repaired"
            ),
            "schema_canonicalization_aggregate_definition": (
                "candidate-level union of all disclosed non-semantic schema operations; "
                "schema_normalization_counts is the per-operation authoritative audit"
            ),
            "control_metadata_source": (
                "immutable runner configuration; model-authored controls are ignored "
                "and their keys are recorded"
            ),
            "control_violations_preserved_before_scoring": True,
            "surrounding_free_text_accepted": False,
            "free_text_wrapping": False,
            "pre_score_processing": config.generation.pre_score_processing,
            "semantic_repair_before_scoring": False,
            "scored_text_field_type_policy": (
                "native JSON strings only; invalid types reject the candidate"
            ),
            "missing_field_policy": (
                "missing required segment keys remain explicit required_fields control "
                "violations; they are not silently inserted as successful authored fields"
            ),
            "control_tag_policy": (
                "native JSON array using only exact `summary` and `salient_point` "
                "tokens; reject case variants, unknown tags, and duplicates without "
                "normalization"
            ),
            "timestamp_type_coercion": False,
            "duplicate_json_key_policy": "reject at every JSON object level",
            "serialized_timestamp_precision": "exact scored float values; no rounding",
            "invalid_timestamp_policy": (
                "reject present nonnumeric, boolean, or nonfinite timestamp fields; "
                "leave numeric range, ordering, coverage, and rhythm to raw scoring"
            ),
            "sampling_method_named_in_generation_prompt": False,
            "source_frame_indices_exposed_to_generator": False,
            "source_duration_exposed_to_generator": False,
            "output_timeline_contract": (
                "compress visible evidence into one 30-second five-segment script"
            ),
            "requested_slots_per_sample": 4,
            "candidate_slot_failure_policy": (
                config.generation.candidate_slot_failure_policy
            ),
            "slot_failure_policy": "continue_remaining_slots",
            "terminal_slot_statuses": ["valid", "failed"],
            "incomplete_pool_policy": (
                "retain each exhausted candidate slot as an explicit failed terminal "
                "slot, preserve all bounded attempts, request every remaining ordered "
                "slot, and serialize the fixed-manifest sample even when zero slots are valid"
            ),
            "failed_sample_replacement": False,
            "invalid_slot_promotion": False,
            "translation_or_semantic_repair_of_invalid_slots": False,
            "method_failure_aggregation": (
                config.generation.method_failure_aggregation
            ),
            "quality_metrics_scope": (
                "report conditional means over samples where a method returns a valid "
                "candidate and failure-aware effective means over all fixed-manifest samples"
            ),
            "effective_failure_values": {
                "reward": 0.0,
                "alignment": 0.0,
                "control_success": 0.0,
                "semantic_point_coverage": 0.0,
                "cider_lite": 0.0,
                "repetition_rate": 1.0,
            },
            "formal_evidence_requirement": (
                "exactly N unique fixed-manifest sample rows, exactly 4N terminal "
                "candidate slots, no unaudited slot, no replacement sample, and no "
                "failed slot promoted as a candidate; explicit terminal failures are "
                "eligible when retained and charged to failure-aware method metrics"
            ),
            "non_english_fallback_accepted": False,
            "translation_fallback": False,
            "degenerate_repetition_policy": (
                "reject repeated-tail responses; the LLaVA service also terminates "
                "pathological token loops early without modifying candidate text"
            ),
            "direct_generation": (
                "C1 when C1 is valid; otherwise an explicit method-level failure"
            ),
            "stars_best_of_4": (
                "argmax full Reward over valid slots in C1-C4; failure only when all "
                "four requested slots are failed"
            ),
        },
        "reward_definition": {
            "configured_weights": configured_weights,
            "normalized_active_weights": {
                key: (value / weight_sum if value > 0.0 else 0.0)
                for key, value in configured_weights.items()
            },
            "control_categories": [
                "segment_count",
                "timestamp_validity",
                "duration_coverage",
                "required_fields",
                "summary_position",
                "information_density",
            ],
            "control_denominator": 6,
            "csr": "one if and only if there are no structural control violations",
            "risk_is_separate_from_control": True,
            "risk_scan_fields": [
                "narration",
                "on_screen_text",
                "salient_point",
                "control_tags",
            ],
            "annotation_dependent_metrics_in_reward": False,
            "alignment_text_fields": [
                "narration",
                "on_screen_text",
                "salient_point",
            ],
        },
        "evaluation_definition": {
            "semantic_point_coverage": {
                "name": "Semantic Point Coverage",
                "abbreviation": "SPC",
                "reference_policy": config.evaluation.semantic_point_reference_policy,
                "reference_construction_protocol": (
                    annotation_reference_protocol.get("name", "")
                ),
                "required_formal_reference_construction_protocol": (
                    FORMAL_REFERENCE_PROTOCOL
                ),
                "reference_construction": annotation_reference_protocol,
                "reference_enters_generation": False,
                "reference_enters_reward": False,
                "encoder": config.evaluation.semantic_point_encoder,
                "similarity": "cosine",
                "similarity_threshold": (
                    config.evaluation.semantic_point_similarity_threshold
                ),
                "script_text_fields": config.evaluation.semantic_point_text_fields,
                "aggregation": (
                    "per sample: fraction of reference semantic points whose maximum "
                    "segment similarity reaches the threshold; dataset mean excludes "
                    "samples without reference points"
                ),
                "missing_reference_value": None,
            }
        },
        "posthoc_analysis": {
            "metric_wise_best_of_4": (
                "per-metric empirical upper bound over C1-C4; columns may come "
                "from different candidates; Reward equals STARS Reward"
            ),
            "uncertainty": "cluster bootstrap by video_id",
        },
        "metric_scale": {
            "reward": "0-1",
            "mv_align": "0-1",
            "semantic_point_coverage": "0-1",
            "cider_lite": "0-1",
            "csr": "0-1",
            "rep": "0-1",
        },
    }


def prepare_protocol_manifest(
    output_dir: str | Path,
    manifest: dict[str, Any],
    resume: bool,
) -> None:
    output = Path(output_dir)
    manifest_path = output / "protocol_manifest.json"
    results_path = output / "results.jsonl"
    if resume and results_path.exists() and results_path.stat().st_size > 0:
        if not manifest_path.exists():
            raise RuntimeError(
                "Refusing to resume results without protocol_manifest.json. "
                "Use a new STARS output directory."
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_fingerprint = existing.get("protocol_fingerprint", "")
        current_fingerprint = manifest.get("protocol_fingerprint", "")
        if existing_fingerprint != current_fingerprint:
            raise RuntimeError(
                "Refusing to mix incompatible experiment rows: protocol fingerprint "
                f"changed from `{existing_fingerprint}` to `{current_fingerprint}`. "
                "Use a new STARS output directory."
            )
        return
    write_json(manifest_path, manifest)
