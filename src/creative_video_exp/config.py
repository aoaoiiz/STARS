from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import read_json, resolve_path


FORMAL_REWARD_WEIGHTS = {
    "alignment": 6.0 / 17.0,
    "readability": 3.0 / 17.0,
    "rhythm": 3.0 / 17.0,
    "control": 4.0 / 17.0,
    "risk": 1.0 / 17.0,
}


@dataclass
class DataConfig:
    name: str = ""
    annotation_path: str = ""
    sample_unit: str = "unique_video"
    video_root: str = ""
    video_roots: list[str] = field(default_factory=list)
    video_search_dirs: list[str] = field(
        default_factory=lambda: ["", "videos", "video", "clips", "all_videos", "clue_video", "clue_videos"]
    )
    split: str = "test"
    offset: int = 0
    limit: int = 0
    hf_dataset: str = ""
    hf_config: str | None = None


@dataclass
class SamplingConfig:
    strategy: str = "binwise"
    num_bins: int = 8
    frames_per_bin: int = 2
    max_frames: int = 16
    synthetic_size: int = 112
    allow_full_decode_fallback: bool = False


@dataclass
class LongVideoConfig:
    enabled: bool = False
    trigger_duration_sec: float = 600.0
    clip_duration_sec: float = 300.0
    merge_similarity_threshold: float = 0.92
    max_clips: int = 0

    def validate(self) -> None:
        if self.trigger_duration_sec <= 0.0:
            raise ValueError("Long-video trigger duration must be positive.")
        if self.clip_duration_sec <= 0.0:
            raise ValueError("Long-video clip duration must be positive.")
        if not 0.0 <= self.merge_similarity_threshold <= 1.0:
            raise ValueError("Long-video merge similarity threshold must be in [0, 1].")
        if self.max_clips < 0:
            raise ValueError("Long-video max_clips must be zero or positive.")


@dataclass
class GenerationConfig:
    num_candidates: int = 4
    target_duration_sec: int = 30
    segments: int = 5
    output_language: str = "English"
    summary_position: str = "late"
    pace: str = "medium"
    information_density: str = "medium"
    low_density_chars_per_second: float = 6.0
    medium_density_chars_per_second: float = 12.0
    high_density_chars_per_second: float = 18.0
    density_length_tolerance: float = 0.5
    pace_duration_profile: list[float] = field(default_factory=list)
    input_protocol: str = "visual_only"
    prompt_version: str = "starsv19_visual_only_contract_guided_v2"
    candidate_generation_protocol: str = "independent_single_candidate_calls"
    parse_retry_count: int = 7
    pre_score_processing: str = (
        "json_envelope_schema_canonicalization_and_timeline_duration_normalization"
    )
    candidate_slot_failure_policy: str = "retain_invalid_slot_and_continue"
    method_failure_aggregation: str = "conditional_and_failure_aware_effective"
    salient_points: list[str] = field(default_factory=list)
    risk_terms: list[str] = field(default_factory=list)

    def density_chars_per_second(self) -> float:
        rates = {
            "low": float(self.low_density_chars_per_second),
            "medium": float(self.medium_density_chars_per_second),
            "high": float(self.high_density_chars_per_second),
        }
        density = self.information_density.strip().lower()
        if density not in rates:
            raise ValueError(
                "information_density must be one of `low`, `medium`, or `high`."
            )
        return rates[density]

    def target_characters_per_segment(self) -> float:
        if self.target_duration_sec <= 0 or self.segments <= 0:
            raise ValueError("Target duration and segment count must be positive.")
        rate = self.density_chars_per_second()
        if rate <= 0:
            raise ValueError("Density rates must be positive.")
        return rate * float(self.target_duration_sec) / self.segments


@dataclass
class RewardConfig:
    alignment_weight: float = FORMAL_REWARD_WEIGHTS["alignment"]
    readability_weight: float = FORMAL_REWARD_WEIGHTS["readability"]
    rhythm_weight: float = FORMAL_REWARD_WEIGHTS["rhythm"]
    control_weight: float = FORMAL_REWARD_WEIGHTS["control"]
    risk_weight: float = FORMAL_REWARD_WEIGHTS["risk"]
    preference_margin: float = 0.08
    visual_grounding_balance: float = 0.5
    text_anchor_semantic_balance: float = 0.7
    readability_balance_beta: float = 0.35
    readability_sentence_bonus: float = 0.08
    rhythm_evenness_balance: float = 0.55
    numerical_epsilon: float = 1e-8

    def component_weights(self) -> dict[str, float]:
        return {
            "alignment": self.alignment_weight,
            "readability": self.readability_weight,
            "rhythm": self.rhythm_weight,
            "control": self.control_weight,
            "risk": self.risk_weight,
        }

    def validate_formal_protocol(self) -> None:
        weights = self.component_weights()
        if any(weight < 0.0 for weight in weights.values()):
            raise ValueError("Formal reward weights must be non-negative.")
        if abs(sum(weights.values()) - 1.0) > 1e-12:
            raise ValueError(
                "STARS reward weights must sum to 1.0; "
                f"received {sum(weights.values()):.17g}."
            )
        if self.readability_balance_beta < 0.0:
            raise ValueError("Readability beta must be non-negative.")
        if not 0.0 <= self.readability_sentence_bonus <= 1.0:
            raise ValueError("Readability sentence bonus must be in [0, 1].")
        if not 0.0 <= self.rhythm_evenness_balance <= 1.0:
            raise ValueError("Rhythm evenness balance must be in [0, 1].")
        if self.numerical_epsilon <= 0.0:
            raise ValueError("Numerical epsilon must be positive.")


@dataclass
class EvaluationConfig:


    semantic_point_coverage_enabled: bool = True
    semantic_point_reference_policy: str = "annotation_only"
    semantic_point_encoder: str = "active_reward_encoder_text_tower"
    semantic_point_similarity_threshold: float = 0.50
    semantic_point_text_fields: list[str] = field(
        default_factory=lambda: ["narration", "on_screen_text", "salient_point"]
    )

    def validate(self) -> None:
        if self.semantic_point_reference_policy != "annotation_only":
            raise ValueError(
                "STARS requires evaluation.semantic_point_reference_policy="
                "`annotation_only`; generated or prompt-provided points are forbidden."
            )
        if self.semantic_point_encoder != "active_reward_encoder_text_tower":
            raise ValueError(
                "STARS requires the frozen active reward encoder text tower for SPC."
            )
        threshold = float(self.semantic_point_similarity_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("SPC cosine-similarity threshold must be in [0, 1].")
        allowed_fields = {"narration", "on_screen_text", "salient_point"}
        fields = list(dict.fromkeys(self.semantic_point_text_fields))
        if not fields or any(field not in allowed_fields for field in fields):
            raise ValueError(
                "SPC text fields must be a non-empty subset of narration, "
                "on_screen_text, and salient_point."
            )
        self.semantic_point_text_fields = fields


@dataclass
class ModelEndpointConfig:
    id: str
    name: str
    role: str
    provider: str = "local"
    adapter: str = ""
    enabled: bool = True
    local_path: str = ""
    endpoint_url: str = ""
    api_key_env: str = ""
    cache_path: str = ""
    device_map: str = "auto"
    dtype: str = "bfloat16"
    quantization: str = ""
    max_frames: int = 16
    max_new_tokens: int = 768
    temperature: float = 0.7
    request_timeout_sec: int = 180
    retry_count: int = 1
    seed: int = 42
    trust_remote_code: bool = True
    checkpoint_identity: dict[str, Any] = field(default_factory=dict)
    checkpoint_manifest_path: str = ""
    notes: str = ""


@dataclass
class ModelSuiteConfig:
    mode: str = "disabled"
    active_video_model: str = ""
    active_reward_vision_model: str = ""
    allow_heavy_model_load: bool = False
    endpoints: list[ModelEndpointConfig] = field(default_factory=list)

    def get(self, endpoint_id: str) -> ModelEndpointConfig | None:
        for endpoint in self.endpoints:
            if endpoint.id == endpoint_id:
                return endpoint
        return None


@dataclass
class ExperimentConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    long_video: LongVideoConfig = field(default_factory=LongVideoConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    models: ModelSuiteConfig = field(default_factory=ModelSuiteConfig)
    output_dir: str = "outputs/stars"
    experiment_version: str = "starsv20"
    config_path: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        payload = _expand_env_values(read_json(path))
        config = cls(
            seed=payload.get("seed", 42),
            data=DataConfig(**payload.get("data", {})),
            sampling=SamplingConfig(**payload.get("sampling", {})),
            long_video=LongVideoConfig(**payload.get("long_video", {})),
            generation=GenerationConfig(**payload.get("generation", {})),
            reward=RewardConfig(**payload.get("reward", {})),
            evaluation=EvaluationConfig(**payload.get("evaluation", {})),
            models=_parse_model_suite(payload.get("models", {})),
            output_dir=payload.get("output_dir", "outputs/stars"),
            experiment_version=payload.get("experiment_version", "starsv20"),
            config_path=str(resolve_path(path)),
        )
        config.evaluation.validate()
        config.long_video.validate()
        return config

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "data": self.data.__dict__,
            "sampling": self.sampling.__dict__,
            "long_video": self.long_video.__dict__,
            "generation": self.generation.__dict__,
            "reward": self.reward.__dict__,
            "evaluation": self.evaluation.__dict__,
            "models": {
                "mode": self.models.mode,
                "active_video_model": self.models.active_video_model,
                "active_reward_vision_model": self.models.active_reward_vision_model,
                "allow_heavy_model_load": self.models.allow_heavy_model_load,
                "endpoints": [endpoint.__dict__ for endpoint in self.models.endpoints],
            },
            "output_dir": self.output_dir,
            "experiment_version": self.experiment_version,
            "config_path": self.config_path,
        }


def _parse_model_suite(payload: dict[str, Any]) -> ModelSuiteConfig:
    if not payload:
        return ModelSuiteConfig()
    endpoints = [
        ModelEndpointConfig(**endpoint_payload)
        for endpoint_payload in payload.get("endpoints", [])
    ]
    return ModelSuiteConfig(
        mode=payload.get("mode", "disabled"),
        active_video_model=payload.get("active_video_model", ""),
        active_reward_vision_model=payload.get("active_reward_vision_model", ""),
        allow_heavy_model_load=payload.get("allow_heavy_model_load", False),
        endpoints=endpoints or ModelSuiteConfig().endpoints,
    )


def _expand_env_values(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*", expanded):
            return ""
        return expanded
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_values(item) for key, item in value.items()}
    return value
