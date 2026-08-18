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
    video_root: str = ""
    video_roots: list[str] = field(default_factory=list)
    video_search_dirs: list[str] = field(
        default_factory=lambda: ["", "videos", "video", "clips", "all_videos", "clue_video", "clue_videos"]
    )
    split: str = "test"
    hf_dataset: str = ""
    hf_config: str | None = None


@dataclass
class SamplingConfig:
    num_bins: int = 8
    frames_per_bin: int = 2
    max_frames: int = 16
    image_size: int = 224


@dataclass
class GenerationConfig:
    num_candidates: int = 4
    target_duration_sec: int = 30
    segments: int = 5
    output_language: str = "English"
    summary_position: str = "late"
    pace: str = "medium"
    information_density: str = "medium"
    target_words_per_segment: int = 12
    min_words_per_segment: int = 6
    max_words_per_segment: int = 18
    input_protocol: str = "visual_only"
    prompt_version: str = "stars_visual_only_fixed_validation_retry_v2"
    candidate_generation_protocol: str = "independent_single_candidate_calls"
    parse_retry_count: int = 7
    pre_score_processing: str = "json_envelope_and_schema_canonicalization_only"
    candidate_slot_failure_policy: str = "retain_invalid_slot_and_continue"
    method_failure_aggregation: str = "conditional_and_failure_aware_effective"
    salient_points: list[str] = field(default_factory=list)
    risk_terms: list[str] = field(default_factory=list)


@dataclass
class RewardConfig:
    alignment_weight: float = FORMAL_REWARD_WEIGHTS["alignment"]
    readability_weight: float = FORMAL_REWARD_WEIGHTS["readability"]
    rhythm_weight: float = FORMAL_REWARD_WEIGHTS["rhythm"]
    control_weight: float = FORMAL_REWARD_WEIGHTS["control"]
    risk_weight: float = FORMAL_REWARD_WEIGHTS["risk"]
    visual_grounding_balance: float = 0.5
    text_anchor_semantic_balance: float = 0.7

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
        if any(
            abs(weights[key] - FORMAL_REWARD_WEIGHTS[key]) > 1e-12
            for key in FORMAL_REWARD_WEIGHTS
        ):
            raise ValueError(
                "STARS requires the fixed composite reward weights "
                "6/17, 3/17, 3/17, 4/17, and 1/17."
            )


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
        if abs(threshold - 0.50) > 1e-12:
            raise ValueError("STARS requires an SPC similarity threshold of 0.50.")
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
    mode: str = "server_full_matrix"
    active_video_model: str = ""
    active_reward_vision_model: str = ""
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
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    models: ModelSuiteConfig = field(default_factory=ModelSuiteConfig)
    output_dir: str = "outputs/full"
    experiment_version: str = "stars"
    config_path: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        payload = _expand_env_values(read_json(path))
        config = cls(
            seed=payload.get("seed", 42),
            data=DataConfig(**payload.get("data", {})),
            sampling=SamplingConfig(**payload.get("sampling", {})),
            generation=GenerationConfig(**payload.get("generation", {})),
            reward=RewardConfig(**payload.get("reward", {})),
            evaluation=EvaluationConfig(**payload.get("evaluation", {})),
            models=_parse_model_suite(payload.get("models", {})),
            output_dir=payload.get("output_dir", "outputs/full"),
            experiment_version=payload.get("experiment_version", "stars"),
            config_path=str(resolve_path(path)),
        )
        config.validate_main_protocol()
        config.evaluation.validate()
        config.reward.validate_formal_protocol()
        return config

    def validate_main_protocol(self) -> None:
        if self.experiment_version != "stars":
            raise ValueError("experiment_version must be `stars`.")
        if self.sampling.num_bins != 8 or self.sampling.frames_per_bin != 2:
            raise ValueError("STARS requires eight temporal bins and two frames per bin.")
        if self.sampling.max_frames != 16:
            raise ValueError("STARS requires a 16-frame input budget.")
        if self.sampling.image_size != 224:
            raise ValueError("STARS requires 224-pixel sampled frames.")
        if self.generation.num_candidates != 4:
            raise ValueError("STARS requires four candidates per sample.")
        if self.generation.output_language != "English":
            raise ValueError("STARS requires English output.")
        word_bounds = (
            self.generation.target_words_per_segment,
            self.generation.min_words_per_segment,
            self.generation.max_words_per_segment,
        )
        if word_bounds != (12, 6, 18):
            raise ValueError("STARS requires target/minimum/maximum word counts of 12/6/18.")
        if self.generation.prompt_version != "stars_visual_only_fixed_validation_retry_v2":
            raise ValueError(
                "STARS requires prompt version "
                "`stars_visual_only_fixed_validation_retry_v2`."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "data": self.data.__dict__,
            "sampling": self.sampling.__dict__,
            "generation": self.generation.__dict__,
            "reward": self.reward.__dict__,
            "evaluation": self.evaluation.__dict__,
            "models": {
                "mode": self.models.mode,
                "active_video_model": self.models.active_video_model,
                "active_reward_vision_model": self.models.active_reward_vision_model,
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
        mode=payload.get("mode", "server_full_matrix"),
        active_video_model=payload.get("active_video_model", ""),
        active_reward_vision_model=payload.get("active_reward_vision_model", ""),
        endpoints=endpoints,
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
