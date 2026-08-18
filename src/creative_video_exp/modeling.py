from __future__ import annotations

import json
import base64
import io
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .config import GenerationConfig, ModelEndpointConfig, ModelSuiteConfig
from .data import VideoSample
from .generation import RUNNER_VARIANT, ScriptCandidate, ScriptSegment
from .representations import Siglip2FrameEncoder
from .utils import normalize_text, sha256_text, stable_hash_int
from .video import SparseFrameBatch


FORMAL_CONTROL_TAGS = frozenset({"summary", "salient_point"})
_TRANSPORT_AUDIT_KEY = "_stars_transport_audit"
_ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class _DuplicateJsonKeyError(ValueError):
    pass


class _TransportRequestError(RuntimeError):
    def __init__(self, message: str, transport_audit: dict[str, Any]):
        super().__init__(message)
        self.transport_audit = transport_audit


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise _DuplicateJsonKeyError(f"duplicate JSON key `{key}`")
        payload[key] = value
    return payload


VALIDATION_RETRY_INSTRUCTION = (
    "\n\nVALIDATION RETRY (fixed STARS protocol): Generate one entirely new "
    "candidate from the supplied video frames. No rejected response or error text "
    "is included in this prompt, and you must not quote, repair, or continue any "
    "earlier response. Return one compact raw JSON object with exactly five timeline "
    "segments and close the object immediately after segment five. Set variant to "
    "the exact ASCII protocol literal visual_story. In narration and salient_point, "
    "describe only visible people, objects, actions, and settings using Latin-script "
    "English. Never quote, transcribe, transliterate, or translate written text from "
    "the frames into variant, narration, or salient_point; never emit bilingual text "
    "or a parenthetical gloss. A brief source-visible title, label, or proper name may "
    "appear verbatim only in on_screen_text; otherwise set on_screen_text to an empty "
    "string. Keep every narration between 6 and 18 English words. Avoid repeated "
    "words, characters, or phrases."
)


class ScriptGenerationModel(Protocol):
    model_report: dict[str, Any]

    def generate(
        self,
        sample: VideoSample,
        batch: SparseFrameBatch,
        config: GenerationConfig,
    ) -> list[ScriptCandidate]:
        ...


class OpenAICompatibleScriptGenerationModel:
    def __init__(self, endpoint: ModelEndpointConfig):
        if endpoint.max_frames != 16:
            raise ValueError("STARS requires exactly 16 input frames.")
        self.endpoint = endpoint
        self.model_report = {
            "runtime": "openai_compatible_multimodal",
            "model_id": endpoint.id,
            "model_name": endpoint.name,
            "provider": endpoint.provider,
            "endpoint_url": endpoint.endpoint_url,
            "max_frames": endpoint.max_frames,
            "adapter": endpoint.adapter,
            "input_protocol": "visual_only",
        }

    def generate(
        self,
        sample: VideoSample,
        batch: SparseFrameBatch,
        config: GenerationConfig,
    ) -> list[ScriptCandidate]:
        if not self.endpoint.endpoint_url:
            raise ValueError("The generation endpoint URL is empty.")
        if config.num_candidates != 4:
            raise ValueError("STARS requires exactly four candidate generations.")
        if config.candidate_generation_protocol != "independent_single_candidate_calls":
            raise ValueError(
                "STARS requires generation.candidate_generation_protocol="
                "`independent_single_candidate_calls`."
            )
        if (
            config.pre_score_processing
            != "json_envelope_and_schema_canonicalization_only"
        ):
            raise ValueError(
                "STARS permits only exact JSON-envelope normalization and schema "
                "canonicalization before reward scoring."
            )
        candidates: list[ScriptCandidate] = []
        semantic_attempt_count = 0
        transport_request_count = 0
        transport_error_count = 0
        parse_failures = 0
        parse_failure_reasons: dict[str, int] = {}
        last_excerpt = ""
        generation_trace: list[dict[str, Any]] = []
        candidate_seconds: list[float] = []
        candidate_usage: list[dict[str, int]] = []
        candidate_server_metrics: list[dict[str, Any]] = []
        slot_seconds: list[float] = []
        slot_usage: list[dict[str, int]] = []
        slot_server_metrics: list[dict[str, Any]] = []
        generation_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        base_prompt = _script_prompt(sample, batch, config)
        retry_prompt = _validation_retry_prompt(base_prompt)
        base_prompt_sha256 = sha256_text(base_prompt)
        retry_prompt_sha256 = sha256_text(retry_prompt)
        max_parse_attempts = max(1, int(config.parse_retry_count) + 1)
        sample_seed_offset = stable_hash_int(sample.video_id or "unknown", 1_000_000)
        request_prompt_hashes: list[str] = []

        for candidate_index in range(1, config.num_candidates + 1):
            slot_trace: dict[str, Any] = {
                "candidate_index": candidate_index,
                "base_prompt_sha256": base_prompt_sha256,
                "validation_retry_prompt_sha256": retry_prompt_sha256,
                "attempts": [],
            }
            accepted: ScriptCandidate | None = None
            for parse_attempt in range(1, max_parse_attempts + 1):
                prompt_kind = (
                    "base_generation" if parse_attempt == 1 else "validation_retry"
                )
                request_prompt = (
                    base_prompt if parse_attempt == 1 else retry_prompt
                )
                request_prompt_sha256 = (
                    base_prompt_sha256
                    if parse_attempt == 1
                    else retry_prompt_sha256
                )
                request_seed = int(
                    self.endpoint.seed
                    + sample_seed_offset
                    + candidate_index * 10_000
                    + parse_attempt - 1
                )
                request_started = time.perf_counter()
                payload = _generation_request_payload(
                    self.endpoint,
                    request_prompt,
                    batch,
                    request_seed,
                )
                try:
                    semantic_attempt_count += 1
                    request_prompt_hashes.append(request_prompt_sha256)
                    response_payload = _post_chat_completion(self.endpoint, payload)
                    request_seconds = time.perf_counter() - request_started
                except RuntimeError as exc:
                    request_seconds = time.perf_counter() - request_started
                    transport_audit = _transport_audit_from_error(
                        exc,
                        fallback_request_seconds=request_seconds,
                    )
                    transport_request_count += int(transport_audit["request_count"])
                    transport_error_count += sum(
                        attempt.get("status") == "error"
                        for attempt in transport_audit["attempts"]
                    )
                    self.model_report["last_error"] = str(exc)
                    slot_trace["attempts"].append(
                        {
                            "attempt": parse_attempt,
                            "prompt_kind": prompt_kind,
                            "prompt_sha256": request_prompt_sha256,
                            "request_seed": request_seed,
                            "request_seconds": round(request_seconds, 6),
                            "status": "endpoint_error",
                            "error": str(exc),
                            "infrastructure_error": True,
                            "usage": dict(_ZERO_USAGE),
                            "server_metrics": {},
                            **_transport_attempt_fields(transport_audit),
                        }
                    )
                    continue

                usage = _usage_counts(response_payload)
                transport_audit = _transport_audit_from_response(
                    response_payload,
                    fallback_request_seconds=request_seconds,
                    success_usage=usage,
                )
                transport_request_count += int(transport_audit["request_count"])
                transport_error_count += sum(
                    attempt.get("status") == "error"
                    for attempt in transport_audit["attempts"]
                )
                server_metrics = _server_metrics(response_payload)
                for key in generation_usage:
                    generation_usage[key] += usage[key]
                content = _generation_response_content(self.endpoint, response_payload)
                last_excerpt = content[:500]
                _, response_envelope = _normalize_json_response_envelope(content)
                candidate, parse_error = _parse_single_script_candidate(
                    content,
                    sample,
                    config,
                    self.endpoint,
                    candidate_index,
                )
                attempt_record = {
                    "attempt": parse_attempt,
                    "prompt_kind": prompt_kind,
                    "prompt_sha256": request_prompt_sha256,
                    "request_seed": request_seed,
                    "request_seconds": round(request_seconds, 6),
                    "response_sha256": sha256_text(content),
                    "raw_response": content,
                    "response_envelope": response_envelope,
                    "json_envelope_normalized": response_envelope
                    == "single_markdown_json_fence",
                    "surrounding_free_text": False,
                    "usage": usage,
                    "server_metrics": server_metrics,
                    **_transport_attempt_fields(transport_audit),
                    "status": "accepted" if candidate is not None else "parse_rejected",
                }
                if parse_error:
                    attempt_record["parse_error"] = parse_error
                slot_trace["attempts"].append(attempt_record)
                if candidate is None:
                    parse_failures += 1
                    reason = parse_error or "unknown parse rejection"
                    parse_failure_reasons[reason] = (
                        parse_failure_reasons.get(reason, 0) + 1
                    )
                    continue

                candidate.candidate_id = _candidate_id(
                    f"cand_{candidate_index:02d}", sample.video_id, candidate_index - 1
                )
                schema_normalizations = list(candidate.schema_normalizations or [])
                schema_envelope_normalized = (
                    "lifted_controls.timeline_to_top_level" in schema_normalizations
                )
                model_authored_control_keys = list(
                    candidate.model_authored_control_keys_ignored or []
                )
                attempt_record["schema_normalizations"] = schema_normalizations
                attempt_record["schema_envelope_normalized"] = (
                    schema_envelope_normalized
                )
                attempt_record["schema_canonicalized"] = bool(schema_normalizations)
                attempt_record["raw_model_variant"] = candidate.raw_model_variant
                attempt_record["model_authored_control_keys_ignored"] = (
                    model_authored_control_keys
                )
                unicode_overlay_audit = _on_screen_text_unicode_audit(candidate)
                attempt_record.update(unicode_overlay_audit)
                candidate.generation_provenance = {
                    "candidate_index": candidate_index,
                    "request_seed": request_seed,
                    "parse_attempt": parse_attempt,
                    "prompt_kind": prompt_kind,
                    "prompt_sha256": request_prompt_sha256,
                    "base_prompt_sha256": base_prompt_sha256,
                    "validation_retry_prompt_sha256": retry_prompt_sha256,
                    "response_sha256": sha256_text(content),
                    "parse_status": "valid_json_candidate",
                    "pre_score_processing": config.pre_score_processing,
                    "semantic_repair_applied": False,
                    "wrapped_free_text": False,
                    "response_envelope": response_envelope,
                    "json_envelope_normalized": response_envelope
                    == "single_markdown_json_fence",
                    "surrounding_free_text": False,
                    "output_language_contract": "english_semantic_fields_unicode_overlay_v2",
                    "variant_policy": "runner_owned_fixed_visual_story",
                    "raw_model_variant": candidate.raw_model_variant,
                    **unicode_overlay_audit,
                    "schema_envelope_normalized": schema_envelope_normalized,
                    "schema_canonicalized": bool(schema_normalizations),
                    "schema_normalizations": schema_normalizations,
                    "model_authored_control_keys_ignored": (
                        model_authored_control_keys
                    ),
                }
                accepted = candidate
                break

            slot_trace["accepted"] = accepted is not None
            one_slot_seconds = sum(
                float(attempt.get("request_seconds", 0.0))
                for attempt in slot_trace["attempts"]
            )
            one_slot_usage = {
                key: sum(
                    int(attempt.get("usage", {}).get(key, 0))
                    for attempt in slot_trace["attempts"]
                )
                for key in generation_usage
            }
            aggregated_server_metrics = _aggregate_slot_server_metrics(
                slot_trace["attempts"]
            )
            slot_transport_request_count = sum(
                int(attempt.get("transport_request_count", 0))
                for attempt in slot_trace["attempts"]
            )
            slot_transport_request_seconds = sum(
                float(attempt.get("transport_request_seconds", 0.0))
                for attempt in slot_trace["attempts"]
            )
            slot_transport_backoff_seconds = sum(
                float(attempt.get("transport_backoff_seconds", 0.0))
                for attempt in slot_trace["attempts"]
            )
            slot_trace.update(
                {
                    "terminal_status": "valid" if accepted is not None else "failed",
                    "terminal_reason": (
                        "accepted_valid_candidate"
                        if accepted is not None
                        else _terminal_slot_failure_reason(slot_trace["attempts"])
                    ),
                    "candidate_id": (
                        accepted.candidate_id if accepted is not None else None
                    ),
                    "request_count": len(slot_trace["attempts"]),
                    "request_seconds": round(one_slot_seconds, 6),
                    "transport_request_count": slot_transport_request_count,
                    "transport_request_seconds": round(
                        slot_transport_request_seconds, 6
                    ),
                    "transport_backoff_seconds": round(
                        slot_transport_backoff_seconds, 6
                    ),
                    "usage": one_slot_usage,
                    "server_metrics": aggregated_server_metrics,
                }
            )
            generation_trace.append(slot_trace)
            slot_seconds.append(round(one_slot_seconds, 6))
            slot_usage.append(one_slot_usage)
            slot_server_metrics.append(aggregated_server_metrics)
            if accepted is None:
                continue
            candidates.append(accepted)
            candidate_seconds.append(round(one_slot_seconds, 6))
            candidate_usage.append(one_slot_usage)
            candidate_server_metrics.append(aggregated_server_metrics)
        self.model_report["generation_attempts"] = semantic_attempt_count
        self.model_report["generation_semantic_attempts"] = semantic_attempt_count
        self.model_report["generation_requests"] = transport_request_count
        self.model_report["generation_transport_requests"] = transport_request_count
        self.model_report["transport_error_count"] = transport_error_count
        self.model_report["last_generation_usage"] = generation_usage
        self.model_report["last_candidate_generation_seconds"] = candidate_seconds
        self.model_report["last_candidate_usage"] = candidate_usage
        self.model_report["last_candidate_server_metrics"] = candidate_server_metrics
        self.model_report["last_slot_generation_seconds"] = slot_seconds
        self.model_report["last_slot_usage"] = slot_usage
        self.model_report["last_slot_server_metrics"] = slot_server_metrics
        self.model_report["last_generation_trace"] = generation_trace
        cumulative_usage = self.model_report.setdefault(
            "cumulative_usage",
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        for key in generation_usage:
            cumulative_usage[key] += generation_usage[key]
        self.model_report["last_prompt_sha256"] = base_prompt_sha256
        self.model_report["base_prompt_sha256"] = base_prompt_sha256
        self.model_report["validation_retry_prompt_sha256"] = retry_prompt_sha256
        self.model_report["prompt_version"] = config.prompt_version
        self.model_report["input_protocol"] = config.input_protocol
        self.model_report["candidate_generation_protocol"] = config.candidate_generation_protocol
        self.model_report["pre_score_processing"] = config.pre_score_processing
        self.model_report["prompt_sha256s"] = request_prompt_hashes
        self.model_report["prompt_templates"] = {
            "base_generation": base_prompt_sha256,
            "validation_retry": retry_prompt_sha256,
        }
        self.model_report["base_seed"] = self.endpoint.seed
        self.model_report["parsed_candidates"] = len(candidates)
        self.model_report["requested_candidate_slots"] = config.num_candidates
        self.model_report["valid_candidate_slots"] = len(candidates)
        self.model_report["failed_candidate_slots"] = (
            config.num_candidates - len(candidates)
        )
        self.model_report["parse_failures"] = parse_failures
        self.model_report["last_parse_failure_reason_counts"] = dict(
            sorted(parse_failure_reasons.items())
        )
        cumulative_reasons = self.model_report.setdefault(
            "cumulative_parse_failure_reason_counts", {}
        )
        for reason, count in parse_failure_reasons.items():
            cumulative_reasons[reason] = int(cumulative_reasons.get(reason, 0)) + count
        if len(candidates) < config.num_candidates:
            self.model_report["last_parse_error"] = (
                f"parsed {len(candidates)} of {config.num_candidates} requested candidates"
            )
            self.model_report["last_response_excerpt"] = last_excerpt
        else:
            self.model_report.pop("last_parse_error", None)
            self.model_report.pop("last_response_excerpt", None)
        return candidates[: config.num_candidates]


def _terminal_slot_failure_reason(attempts: list[dict[str, Any]]) -> str:
    statuses = {str(attempt.get("status", "")) for attempt in attempts}
    if statuses == {"endpoint_error"}:
        return "bounded_endpoint_attempts_exhausted"
    if statuses == {"parse_rejected"}:
        return "bounded_validation_attempts_exhausted"
    return "bounded_mixed_attempts_exhausted"


def _transport_attempt_fields(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "transport_attempts": [dict(item) for item in audit["attempts"]],
        "transport_request_count": int(audit["request_count"]),
        "transport_request_seconds": round(float(audit["request_seconds"]), 6),
        "transport_backoff_seconds": round(float(audit["backoff_seconds"]), 6),
    }


def _transport_audit_from_response(
    response_payload: dict[str, Any],
    fallback_request_seconds: float,
    success_usage: dict[str, int],
) -> dict[str, Any]:
    raw = response_payload.pop(_TRANSPORT_AUDIT_KEY, None)
    if isinstance(raw, dict) and isinstance(raw.get("attempts"), list) and raw["attempts"]:
        return raw
    return {
        "attempts": [
            {
                "transport_attempt": 1,
                "status": "success",
                "request_seconds": round(float(fallback_request_seconds), 6),
                "backoff_seconds_after": 0.0,
                "usage": dict(success_usage),
            }
        ],
        "request_count": 1,
        "request_seconds": round(float(fallback_request_seconds), 6),
        "backoff_seconds": 0.0,
    }


def _transport_audit_from_error(
    error: RuntimeError,
    fallback_request_seconds: float,
) -> dict[str, Any]:
    raw = getattr(error, "transport_audit", None)
    if isinstance(raw, dict) and isinstance(raw.get("attempts"), list) and raw["attempts"]:
        return raw
    return {
        "attempts": [
            {
                "transport_attempt": 1,
                "status": "error",
                "request_seconds": round(float(fallback_request_seconds), 6),
                "backoff_seconds_after": 0.0,
                "error_type": type(error).__name__,
                "error": str(error),
                "usage": dict(_ZERO_USAGE),
            }
        ],
        "request_count": 1,
        "request_seconds": round(float(fallback_request_seconds), 6),
        "backoff_seconds": 0.0,
    }


def _aggregate_slot_server_metrics(
    semantic_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [
        dict(attempt.get("server_metrics", {}))
        for attempt in semantic_attempts
        if attempt.get("server_metrics")
    ]
    if not rows:
        return {}
    aggregated = dict(rows[-1])
    for key in ("peak_allocated_mib", "peak_reserved_mib"):
        values = [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
        ]
        if values:
            aggregated[key] = max(values)
    if any("degenerate_repetition_early_stop" in row for row in rows):
        aggregated["degenerate_repetition_early_stop"] = any(
            row.get("degenerate_repetition_early_stop") is True for row in rows
        )
    aggregated["server_metrics_response_count"] = len(rows)
    aggregated["server_metrics_aggregation"] = (
        "last_response_metadata_with_slot_max_peaks_and_any_events"
    )
    return aggregated


def build_frame_encoder(model_suite: ModelSuiteConfig) -> tuple[Any, dict[str, Any]]:
    requested_encoder_id = (
        model_suite.active_reward_vision_model or model_suite.active_video_model
    )
    endpoint = model_suite.get(requested_encoder_id)
    if endpoint is None:
        raise ValueError(f"Reward vision model `{requested_encoder_id}` was not found.")
    if not endpoint.enabled:
        raise ValueError(f"Reward vision model `{endpoint.id}` is disabled.")
    if endpoint.adapter != "siglip2_frame_encoder":
        raise ValueError(
            "STARS requires the `siglip2_frame_encoder` reward adapter; "
            f"received `{endpoint.adapter}`."
        )
    encoder = Siglip2FrameEncoder(
        model_path=endpoint.local_path,
        device=endpoint.device_map,
        dtype=endpoint.dtype,
    )
    encoder.model_report["model_id"] = endpoint.id
    return encoder, dict(encoder.model_report)


def build_script_generation_model(model_suite: ModelSuiteConfig) -> ScriptGenerationModel:
    endpoint = model_suite.get(model_suite.active_video_model)
    if endpoint is None:
        raise ValueError(
            f"Generation model `{model_suite.active_video_model}` was not found."
        )
    if not endpoint.enabled:
        raise ValueError(f"Generation model `{endpoint.id}` is disabled.")
    if endpoint.provider not in {"openai_compatible", "openai"}:
        raise ValueError(
            "STARS requires an OpenAI-compatible multimodal generation endpoint; "
            f"received provider `{endpoint.provider}`."
        )
    return OpenAICompatibleScriptGenerationModel(endpoint)


def _script_prompt(sample: VideoSample, batch: SparseFrameBatch, config: GenerationConfig) -> str:
    if config.input_protocol != "visual_only":
        raise ValueError(
            "STARS supports only `visual_only`; annotation context "
            "must remain evaluation-only."
        )
    return (
        "You are a multimodal structured timeline script generator for ordinary videos. "
        "Generate structured English timeline scripts grounded only in the provided video frames. "
        "All narration and salient_point values must be written in English. "
        "Set the top-level variant field to the exact ASCII protocol literal "
        "`visual_story`; variant is a fixed runner-owned schema label, not a place to "
        "describe the video or copy visible text. "
        "Common Unicode punctuation, symbols, units, and Latin-script names are allowed. "
        "The `salient_point` field records important visible information, not an advertising claim. "
        "Never emit bilingual text or parenthetical translations. "
        "Some frames may visibly contain burned-in subtitles, captions, interface text, signs, or OCR in another language. "
        "Do not copy such text into narration or salient_point; describe the associated visible event in original English words. "
        "The optional on_screen_text field may preserve a brief source-visible title, label, or proper name in its original "
        "Unicode script when it is clearly visible; otherwise use an empty string. Do not translate or invent overlay text. "
        "English brand names may be retained when visually supported. Before returning the JSON, silently verify that "
        "narration and salient_point use Latin-script English and variant is exactly `visual_story`. "
        "Hard language boundary: Chinese, Japanese, Korean, Cyrillic, and other non-Latin letters are forbidden in "
        "narration and salient_point. If a frame contains such text, describe the visible event using original "
        "English wording; only a brief source-visible title, label, or proper name may remain in on_screen_text. "
        "Do not invent claims unsupported by the visible evidence. "
        "The attached images are sparse observations shown in chronological order. "
        "They are visual evidence, not one-to-one output segments, timestamps, or a statement "
        "of the source-video duration. Do not describe every frame or reconstruct the full "
        "source-video timeline. Instead, compress the visible story into one new 30-second script. "
        "Questions, answers, annotation captions, transcripts, metadata descriptions, and reference text "
        "are evaluation-only and are not provided as textual prompt context. "
        "Return ONLY one valid JSON object, with no array, markdown, explanation, or surrounding text. "
        "Start the response with { and end it with }; never use ```json code fences. "
        "The only top-level fields are candidate_id, variant, and timeline. "
        "Do not output a controls object; the experiment runner attaches fixed control metadata. "
        "The timeline array must be a direct top-level field. Never place timeline inside controls, "
        "summary, script, output, or any other wrapper object. "
        "The timeline must contain exactly five objects, no more and no fewer, regardless of how many "
        "input frames are provided. Never create one timeline segment per input frame. Before returning, "
        "silently count the timeline objects and verify that the count is exactly five. "
        "Each timeline object must explicitly contain all six keys: start, end, narration, on_screen_text, "
        "salient_point, and control_tags. Narration must be non-empty. on_screen_text and salient_point may "
        "be empty strings when the visible evidence does not support them. control_tags must be a JSON array "
        "and may be empty for non-summary segments. "
        f"Use start=0 for the first item and end={config.target_duration_sec} for the last item. "
        f"Keep every timestamp within 0-{config.target_duration_sec} seconds, with increasing, "
        f"non-overlapping segments that collectively cover the full {config.target_duration_sec} seconds. "
        "Choose the internal boundaries according to the visible narrative rhythm rather than copying frame indices. "
        "Use the exact control tag `summary` exactly once, only in the fifth and final segment. Do not use "
        "the summary tag in any earlier segment. Use `salient_point` only when a segment carries important "
        "visible information. "
        f"Generate exactly one candidate; target duration {config.target_duration_sec}s; "
        f"{config.segments} timeline segments; summary position {config.summary_position}; "
        f"pace {config.pace}; information density {config.information_density}. "
        f"Aim for about {config.target_words_per_segment} English words per narration segment "
        f"(acceptable range {config.min_words_per_segment}-{config.max_words_per_segment}). "
        f"Risk terms to avoid: {config.risk_terms}."
    )


def _validation_retry_prompt(base_prompt: str) -> str:
    return base_prompt + VALIDATION_RETRY_INSTRUCTION


def _request_headers(endpoint: ModelEndpointConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key_env:
        api_key = os.environ.get(endpoint.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_chat_completion(endpoint: ModelEndpointConfig, payload: dict[str, Any]) -> dict[str, Any]:
    attempts = max(1, int(endpoint.retry_count or 0) + 1)
    last_error: Exception | None = None
    transport_attempts: list[dict[str, Any]] = []
    total_backoff_seconds = 0.0
    for attempt in range(attempts):
        request = urllib.request.Request(
            endpoint.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=_request_headers(endpoint),
            method="POST",
        )
        transport_started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=endpoint.request_timeout_sec) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            request_seconds = time.perf_counter() - transport_started
            usage = _usage_counts(response_payload)
            transport_attempts.append(
                {
                    "transport_attempt": attempt + 1,
                    "status": "success",
                    "request_seconds": round(request_seconds, 6),
                    "backoff_seconds_after": 0.0,
                    "usage": usage,
                }
            )
            response_payload[_TRANSPORT_AUDIT_KEY] = {
                "attempts": transport_attempts,
                "request_count": len(transport_attempts),
                "request_seconds": round(
                    sum(float(item["request_seconds"]) for item in transport_attempts),
                    6,
                ),
                "backoff_seconds": round(total_backoff_seconds, 6),
            }
            return response_payload
        except (
            urllib.error.URLError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            request_seconds = time.perf_counter() - transport_started
            last_error = exc
            record = {
                "transport_attempt": attempt + 1,
                "status": "error",
                "request_seconds": round(request_seconds, 6),
                "backoff_seconds_after": 0.0,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "usage": dict(_ZERO_USAGE),
            }
            if attempt + 1 < attempts:
                backoff_started = time.perf_counter()
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
                backoff_seconds = time.perf_counter() - backoff_started
                record["backoff_seconds_after"] = round(backoff_seconds, 6)
                total_backoff_seconds += backoff_seconds
            transport_attempts.append(record)
    audit = {
        "attempts": transport_attempts,
        "request_count": len(transport_attempts),
        "request_seconds": round(
            sum(float(item["request_seconds"]) for item in transport_attempts), 6
        ),
        "backoff_seconds": round(total_backoff_seconds, 6),
    }
    raise _TransportRequestError(
        str(last_error) if last_error else "unknown endpoint error",
        audit,
    )


def _message_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(normalize_text(item.get("text", item.get("content", ""))))
            else:
                parts.append(normalize_text(item))
        return "\n".join(part for part in parts if part)
    return normalize_text(content)


def _frame_content(batch: SparseFrameBatch, max_frames: int) -> list[dict[str, Any]]:
    frames = batch.frames[: max(1, max_frames)]
    content = []
    for frame in frames:
        image = Image.fromarray(frame).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
    return content


def _generation_request_payload(
    endpoint: ModelEndpointConfig,
    prompt: str,
    batch: SparseFrameBatch,
    seed: int,
) -> dict[str, Any]:
    if endpoint.adapter == "openai_responses_multimodal":
        images = [
            {
                "type": "input_image",
                "image_url": item["image_url"]["url"],
            }
            for item in _frame_content(batch, endpoint.max_frames)
        ]
        return {
            "model": endpoint.name,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        *images,
                    ],
                }
            ],
            "max_output_tokens": endpoint.max_new_tokens,
        }
    return {
        "model": endpoint.name,
        "temperature": endpoint.temperature,
        "max_tokens": endpoint.max_new_tokens,
        "seed": seed,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *_frame_content(batch, endpoint.max_frames),
                ],
            }
        ],
    }


def _generation_response_content(
    endpoint: ModelEndpointConfig,
    response_payload: dict[str, Any],
) -> str:
    if endpoint.adapter != "openai_responses_multimodal":
        return _message_content(response_payload)
    parts: list[str] = []
    for output_item in response_payload.get("output", []):
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content", []):
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") == "output_text":
                parts.append(normalize_text(content_item.get("text", "")))
    return "\n".join(part for part in parts if part)


def _usage_counts(response_payload: dict[str, Any]) -> dict[str, int]:
    usage = response_payload.get("usage") or {}
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(
        usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    )
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _server_metrics(response_payload: dict[str, Any]) -> dict[str, Any]:
    payload = response_payload.get("server_metrics") or {}
    if not isinstance(payload, dict):
        return {}
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[str(key)] = value
    returned_model_id = response_payload.get("model")
    if isinstance(returned_model_id, str) and returned_model_id.strip():
        returned_model_id = returned_model_id.strip()
        existing = clean.get("returned_model_id")
        if isinstance(existing, str) and existing and existing != returned_model_id:
            clean["server_metrics_returned_model_id"] = existing
            clean["returned_model_id_conflict"] = True
        clean["returned_model_id"] = returned_model_id
    return clean


def _parse_script_candidates(
    content: str,
    sample: VideoSample,
    config: GenerationConfig,
    endpoint: ModelEndpointConfig | None = None,
) -> list[ScriptCandidate]:
    candidates: list[ScriptCandidate] = []
    normalized_content, _ = _normalize_json_response_envelope(content)
    try:
        payload = json.loads(
            normalized_content,
            object_pairs_hook=_strict_json_object,
        )
    except (json.JSONDecodeError, _DuplicateJsonKeyError):
        return []
    payload_is_array = isinstance(payload, list)
    items = payload if payload_is_array else [payload]
    for candidate_index, item in enumerate(items, start=1):
        candidate, _ = _parse_single_script_candidate(
            json.dumps(item) if payload_is_array else normalized_content,
            sample,
            config,
            endpoint,
            candidate_index,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _parse_single_script_candidate(
    content: str,
    sample: VideoSample,
    config: GenerationConfig,
    endpoint: ModelEndpointConfig | None,
    candidate_index: int,
) -> tuple[ScriptCandidate | None, str]:
    repetition_issue = _degenerate_repetition_issue(content)
    if repetition_issue:
        return None, repetition_issue
    normalized_content, response_envelope = _normalize_json_response_envelope(content)
    try:
        payload = json.loads(
            normalized_content,
            object_pairs_hook=_strict_json_object,
        )
    except _DuplicateJsonKeyError as exc:
        return None, f"response contains an ambiguous {exc}"
    except json.JSONDecodeError as exc:
        if response_envelope == "single_markdown_json_fence":
            return None, f"fenced response body is not valid JSON: {exc.msg}"
        return None, f"response is not a raw JSON object: {exc.msg}"
    if isinstance(payload, list):
        return None, f"expected one JSON object, received an array of {len(payload)} items"
    if not isinstance(payload, dict):
        return None, "top-level JSON value is not an object"

    schema_normalizations: list[str] = []
    raw_controls = payload.get("controls")
    model_authored_control_keys = (
        sorted(str(key) for key in raw_controls)
        if isinstance(raw_controls, dict)
        else []
    )
    raw_timeline = payload.get("timeline")
    lifted_nested_timeline = False
    if not isinstance(raw_timeline, list):
        nested_timeline = (
            raw_controls.get("timeline")
            if isinstance(raw_controls, dict)
            else None
        )
        if not isinstance(nested_timeline, list):
            return None, "the required `timeline` field is not a JSON array"
        raw_timeline = nested_timeline
        lifted_nested_timeline = True
        schema_normalizations.append("lifted_controls.timeline_to_top_level")
    ignored_model_control_keys = [
        key
        for key in model_authored_control_keys
        if not (lifted_nested_timeline and key == "timeline")
    ]
    if ignored_model_control_keys:
        schema_normalizations.append("ignored_model_authored_controls")

    parse_issues: list[str] = []
    missing_required_fields: list[str] = []
    required_segment_keys = (
        "start",
        "end",
        "narration",
        "on_screen_text",
        "salient_point",
        "control_tags",
    )
    timeline: list[ScriptSegment] = []
    for segment_index, raw_segment in enumerate(raw_timeline, start=1):
        if not isinstance(raw_segment, dict):
            return None, f"timeline[{segment_index}] is not a JSON object"
        for key in required_segment_keys:
            if key not in raw_segment:
                field_path = f"timeline[{segment_index}].{key}"
                missing_required_fields.append(field_path)
                parse_issues.append(f"{field_path} is missing")

        semantic_values: dict[str, str] = {}
        for key in ("narration", "on_screen_text", "salient_point"):
            if key not in raw_segment:
                semantic_values[key] = ""
                continue
            value = raw_segment[key]
            if not isinstance(value, str):
                return None, (
                    f"timeline[{segment_index}].{key} must be a JSON string; "
                    f"received {type(value).__name__}"
                )
            semantic_values[key] = value

        raw_control_tags = raw_segment.get("control_tags", [])
        if "control_tags" in raw_segment:
            if not isinstance(raw_control_tags, list):
                return None, (
                    f"timeline[{segment_index}].control_tags must be a JSON array; "
                    f"received {type(raw_control_tags).__name__}"
                )
            non_string_tag_index = next(
                (
                    tag_index
                    for tag_index, tag in enumerate(raw_control_tags, start=1)
                    if not isinstance(tag, str)
                ),
                None,
            )
            if non_string_tag_index is not None:
                return None, (
                    f"timeline[{segment_index}].control_tags[{non_string_tag_index}] "
                    "must be a JSON string"
                )
            invalid_tags = [
                tag for tag in raw_control_tags if tag not in FORMAL_CONTROL_TAGS
            ]
            if invalid_tags:
                return None, (
                    f"timeline[{segment_index}].control_tags contains a noncanonical "
                    f"tag {invalid_tags[0]!r}; allowed tags are `summary` and "
                    "`salient_point`"
                )
            if len(raw_control_tags) != len(set(raw_control_tags)):
                return None, (
                    f"timeline[{segment_index}].control_tags contains duplicate tags"
                )
        control_tags = list(raw_control_tags)

        for key in ("start", "end"):
            if key not in raw_segment:
                continue
            value = raw_segment[key]
            if (
                not _is_finite_json_number(value)
            ):
                return None, (
                    f"timeline[{segment_index}].{key} must be a finite JSON number; "
                    f"received {type(value).__name__}"
                )

        start = _optional_float(raw_segment.get("start"))
        end = _optional_float(raw_segment.get("end"))
        if start is None:
            parse_issues.append(
                f"timeline[{segment_index}].start is not a finite JSON number"
            )
        if end is None:
            parse_issues.append(
                f"timeline[{segment_index}].end is not a finite JSON number"
            )
        timeline.append(
            ScriptSegment(
                start=start,
                end=end,
                narration=semantic_values["narration"],
                on_screen_text=semantic_values["on_screen_text"],
                salient_point=semantic_values["salient_point"],
                control_tags=control_tags,
            )
        )

    controls = _runner_control_metadata(config, endpoint)
    raw_model_variant = normalize_text(payload.get("variant", ""))
    if raw_model_variant != RUNNER_VARIANT:
        schema_normalizations.append("fixed_runner_owned_variant")

    candidate = ScriptCandidate(
        candidate_id=_candidate_id(
            payload.get("candidate_id", ""), sample.video_id, candidate_index - 1
        ),
        timeline=timeline,
        controls=controls,
        text="\n".join(segment.narration for segment in timeline),
        variant=RUNNER_VARIANT,
        raw_model_variant=raw_model_variant,
        parse_issues=parse_issues,
        missing_required_fields=missing_required_fields,
        schema_normalizations=schema_normalizations,
        model_authored_control_keys_ignored=ignored_model_control_keys,
    )
    if config.output_language.strip().lower() == "english":
        contract_issue = _english_unicode_contract_issue(candidate)
        if contract_issue:
            return None, (
                "candidate violates the English semantic-field contract: "
                f"{contract_issue}"
            )
    return candidate, ""


def _degenerate_repetition_issue(content: str) -> str:
    value = content.rstrip()
    if len(value) < 256:
        return ""
    repeat_count = 8
    for period in range(1, 25):
        required = period * repeat_count
        if len(value) < required:
            continue
        repeated_tail = value[-required:]
        pattern = repeated_tail[-period:]
        if not any(character.isalpha() for character in pattern):
            continue
        if repeated_tail == pattern * repeat_count:
            return (
                "response ended in degenerate repetition: "
                f"period={period} characters, repeats>={repeat_count}"
            )
    return ""


def _runner_control_metadata(
    config: GenerationConfig,
    endpoint: ModelEndpointConfig | None,
) -> dict[str, Any]:
    return {
        "target_duration_sec": config.target_duration_sec,
        "segments": config.segments,
        "output_language": config.output_language,
        "summary_position": config.summary_position,
        "pace": config.pace,
        "information_density": config.information_density,
        "target_words_per_segment": config.target_words_per_segment,
        "min_words_per_segment": config.min_words_per_segment,
        "max_words_per_segment": config.max_words_per_segment,
        "salient_points": list(config.salient_points),
        "risk_terms": list(config.risk_terms),
        "_generation_runtime": {
            "generated_by": "openai_compatible_multimodal",
            "model_id": endpoint.id if endpoint else "",
            "model_name": endpoint.name if endpoint else "",
        },
    }


def _english_unicode_contract_issue(candidate: ScriptCandidate) -> str:
    fields: list[tuple[str, str]] = []
    for index, segment in enumerate(candidate.timeline, start=1):
        fields.extend(
            [
                (f"timeline[{index}].narration", segment.narration),
                (f"timeline[{index}].salient_point", segment.salient_point),
            ]
        )
    fields.extend(_control_string_fields(candidate.controls))

    for path, value in fields:
        for character in value:
            if not character.isalpha():
                continue
            unicode_name = unicodedata.name(character, "")
            if "LATIN" not in unicode_name:
                return (
                    f"{path} contains non-Latin letter "
                    f"U+{ord(character):04X} ({unicode_name or 'UNKNOWN'})"
                )
    return ""


def _on_screen_text_unicode_audit(candidate: ScriptCandidate) -> dict[str, Any]:
    segment_indices: list[int] = []
    non_latin_letter_count = 0
    total_letter_count = 0
    for index, segment in enumerate(candidate.timeline, start=1):
        segment_has_non_latin = False
        for character in segment.on_screen_text:
            if not character.isalpha():
                continue
            total_letter_count += 1
            unicode_name = unicodedata.name(character, "")
            if "LATIN" not in unicode_name:
                non_latin_letter_count += 1
                segment_has_non_latin = True
        if segment_has_non_latin:
            segment_indices.append(index)
    return {
        "on_screen_text_has_non_latin": bool(non_latin_letter_count),
        "on_screen_text_non_latin_segment_indices": segment_indices,
        "on_screen_text_non_latin_letter_count": non_latin_letter_count,
        "on_screen_text_total_letter_count": total_letter_count,
    }


def _control_string_fields(
    value: Any,
    path: str = "controls",
) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        rows: list[tuple[str, str]] = []
        for key, nested in value.items():
            if str(key).startswith("_"):
                continue
            rows.extend(_control_string_fields(nested, f"{path}.{key}"))
        return rows
    if isinstance(value, (list, tuple)):
        rows = []
        for index, nested in enumerate(value):
            rows.extend(_control_string_fields(nested, f"{path}[{index}]"))
        return rows
    return []


_SINGLE_JSON_FENCE = re.compile(
    r"\A\s*```json\s*(?P<body>.*?)\s*```\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def _normalize_json_response_envelope(content: str) -> tuple[str, str]:
    match = _SINGLE_JSON_FENCE.fullmatch(content)
    if match is None:
        return content, "raw_json"
    return match.group("body").strip(), "single_markdown_json_fence"


def _optional_float(value: Any) -> float | None:
    if not _is_finite_json_number(value):
        return None
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError):
        return None


def _is_finite_json_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (OverflowError, TypeError, ValueError):
        return False


def _candidate_id(raw_id: Any, video_id: str, idx: int) -> str:
    raw = normalize_text(raw_id) or f"vlm_{idx:02d}"
    video = normalize_text(video_id) or "video"
    if not raw.startswith(video):
        raw = f"{video}_{raw}"
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("_") or f"{video}_vlm_{idx:02d}"
