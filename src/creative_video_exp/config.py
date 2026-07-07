from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import read_json, resolve_path


@dataclass
class DataConfig:
    name: str = "smoke"
    annotation_path: str = ""
    video_root: str = ""
    video_roots: list[str] = field(default_factory=list)
    video_search_dirs: list[str] = field(
        default_factory=lambda: ["", "videos", "video", "clips", "all_videos", "clue_video", "clue_videos"]
    )
    split: str = "test"
    offset: int = 0
    limit: int = 8
    hf_dataset: str = ""
    hf_config: str | None = None


@dataclass
class SamplingConfig:
    num_bins: int = 8
    frames_per_bin: int = 2
    max_frames: int = 16
    synthetic_size: int = 112


@dataclass
class GenerationConfig:
    num_candidates: int = 6
    target_duration_sec: int = 30
    segments: int = 5
    cta_position: str = "late"
    pace: str = "medium"
    information_density: str = "medium"
    selling_points: list[str] = field(
        default_factory=lambda: ["核心卖点", "使用场景", "信任背书"]
    )
    risk_terms: list[str] = field(default_factory=list)


@dataclass
class RewardConfig:
    alignment_weight: float = 0.35
    readability_weight: float = 0.2
    rhythm_weight: float = 0.2
    control_weight: float = 0.2
    risk_weight: float = 0.05
    text_reasoning_weight: float = 0.0
    preference_margin: float = 0.08


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
    trust_remote_code: bool = True
    notes: str = ""


@dataclass
class ModelSuiteConfig:
    mode: str = "local_smoke"
    active_video_model: str = "local_stats"
    active_text_reward_model: str = ""
    allow_heavy_model_load: bool = False
    endpoints: list[ModelEndpointConfig] = field(
        default_factory=lambda: [
            ModelEndpointConfig(
                id="local_stats",
                name="StatsFrameEncoder",
                role="video_representation",
                provider="local",
                adapter="stats_frame_encoder",
                notes="Lightweight deterministic smoke-test encoder.",
            )
        ]
    )

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
    models: ModelSuiteConfig = field(default_factory=ModelSuiteConfig)
    output_dir: str = "outputs/smoke"
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
            models=_parse_model_suite(payload.get("models", {})),
            output_dir=payload.get("output_dir", "outputs/smoke"),
            config_path=str(resolve_path(path)),
        )
        return config

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "data": self.data.__dict__,
            "sampling": self.sampling.__dict__,
            "generation": self.generation.__dict__,
            "reward": self.reward.__dict__,
            "models": {
                "mode": self.models.mode,
                "active_video_model": self.models.active_video_model,
                "active_text_reward_model": self.models.active_text_reward_model,
                "allow_heavy_model_load": self.models.allow_heavy_model_load,
                "endpoints": [endpoint.__dict__ for endpoint in self.models.endpoints],
            },
            "output_dir": self.output_dir,
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
        mode=payload.get("mode", "local_smoke"),
        active_video_model=payload.get("active_video_model", "local_stats"),
        active_text_reward_model=payload.get("active_text_reward_model", ""),
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
