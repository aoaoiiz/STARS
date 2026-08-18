from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_identity import checkpoint_identity_summary, load_checkpoint_manifest
from .utils import cosine
from .video import SparseFrameBatch


@dataclass
class VideoRepresentation:
    frame_embeddings: np.ndarray
    video_embedding: np.ndarray
    keyframe_local_indices: list[int]
    keyframe_source_indices: list[int]
    aggregation_weights: list[float]
    content_tags: list[str]
    text_anchor: str
    diagnostics: dict[str, float | str] = field(default_factory=dict)
    text_anchor_embedding: np.ndarray | None = field(default=None, repr=False)
    alignment_backend: Any | None = field(default=None, repr=False)


class Siglip2FrameEncoder:
    TAG_GROUPS = {
        "scene": [
            ("indoor scene", "a photo of an indoor scene"),
            ("outdoor scene", "a photo of an outdoor scene"),
            ("home interior", "a photo of a home interior"),
            ("office", "a photo of an office"),
            ("retail store", "a photo of a retail store"),
            ("street", "a photo of a street"),
            ("natural landscape", "a photo of a natural landscape"),
            ("sports venue", "a photo of a sports venue"),
        ],
        "object": [
            ("person", "a photo containing a person"),
            ("vehicle", "a photo containing a vehicle"),
            ("food", "a photo containing food"),
            ("beverage", "a photo containing a beverage"),
            ("electronic device", "a photo containing an electronic device"),
            ("furniture", "a photo containing furniture"),
            ("clothing", "a photo containing clothing"),
            ("animal", "a photo containing an animal"),
            ("building", "a photo containing a building"),
            ("product package", "a photo containing a product package"),
        ],
        "activity": [
            ("talking", "a photo of a person talking"),
            ("walking", "a photo of a person walking"),
            ("cooking", "a photo of someone cooking"),
            ("driving", "a photo of someone driving"),
            ("using a device", "a photo of someone using an electronic device"),
            ("presenting a product", "a photo of someone presenting a product"),
            ("eating", "a photo of someone eating"),
            ("playing sports", "a photo of someone playing sports"),
            ("shopping", "a photo of someone shopping"),
            ("working", "a photo of someone working"),
        ],
    }
    TAGS_PER_GROUP = {"scene": 1, "object": 2, "activity": 1}

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        centrality_weight: float = 0.55,
        aggregation_temperature: float = 0.35,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "SigLIP2 reward encoding requires torch and transformers. "
                "Install the formal reward environment before running the experiment."
            ) from exc

        resolved_path = Path(model_path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"SigLIP2 checkpoint directory not found: {resolved_path}")

        self.torch = torch
        self.model_path = str(resolved_path)
        self.device = self._resolve_device(device)
        self.dtype_name = dtype
        self.dtype = self._resolve_dtype(dtype)
        self.centrality_weight = float(centrality_weight)
        self.aggregation_temperature = float(aggregation_temperature)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=False,
        )
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
        }
        if self.device != "cpu":
            load_kwargs["dtype"] = self.dtype
        self.model = AutoModel.from_pretrained(self.model_path, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.tag_labels, tag_prompts, self.tag_group_slices = self._tag_taxonomy()
        self.tag_embeddings = self._encode_texts(tag_prompts)
        checkpoint_identity: dict[str, Any] = {}
        checkpoint_manifest_path = os.environ.get(
            "REWARD_VISION_CHECKPOINT_MANIFEST",
            "",
        ).strip()
        if checkpoint_manifest_path:
            checkpoint_manifest = load_checkpoint_manifest(checkpoint_manifest_path)
            checkpoint_identity = checkpoint_identity_summary(checkpoint_manifest)
            if checkpoint_identity["model_id"] != "google/siglip2-so400m-patch14-384":
                raise RuntimeError(
                    "Reward checkpoint manifest model_id does not match formal SigLIP2."
                )
        self.model_report = {
            "runtime": "frozen_siglip2_vision_text_reward",
            "model_name": "google/siglip2-so400m-patch14-384",
            "provider": "huggingface_local",
            "adapter": "siglip2_frame_encoder",
            "local_path": self.model_path,
            "checkpoint_config_sha256": self._config_sha256(),
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_identity_sha256": checkpoint_identity.get(
                "identity_sha256", ""
            ),
            "device": self.device,
            "dtype": self.dtype_name if self.device != "cpu" else "float32",
            "embedding_dim": int(self.tag_embeddings.shape[1]),
            "centrality_weight_alpha": self.centrality_weight,
            "aggregation_temperature_gamma": self.aggregation_temperature,
            "tag_vocabulary_version": "coarse_scene_object_activity_v1",
            "tag_vocabulary_size": len(self.tag_labels),
            "image_processor_fast": False,
            "text_max_length": 64,
            "frozen": True,
        }

    def encode(self, batch: SparseFrameBatch, context_text: str = "") -> VideoRepresentation:
        frame_embeddings = self._encode_images(batch.frames)
        weights = _aggregation_weights(
            frame_embeddings,
            centrality_weight=self.centrality_weight,
            temperature=self.aggregation_temperature,
        )
        video_embedding = _l2_normalize(
            (frame_embeddings * weights[:, None]).sum(axis=0, keepdims=True)
        )[0]
        key_local = np.argsort(weights)[::-1][: min(4, len(weights))].tolist()
        key_local = sorted(int(index) for index in key_local)
        key_source = [batch.selected_indices[index] for index in key_local]
        tags = self._select_tags(video_embedding)
        text_anchor = " ".join([*tags, context_text]).strip()
        text_anchor_embedding = self._encode_texts([text_anchor or "visual evidence"])[0]
        diagnostics = {
            "source_kind": batch.source_kind,
            "frame_encoder_runtime": "frozen_siglip2_vision_text_reward",
            "frame_encoder_model": "google/siglip2-so400m-patch14-384",
            "embedding_dim": float(frame_embeddings.shape[1]),
            "num_encoded_frames": float(len(frame_embeddings)),
            "tag_vocabulary_version": "coarse_scene_object_activity_v1",
        }
        return VideoRepresentation(
            frame_embeddings=frame_embeddings,
            video_embedding=video_embedding,
            keyframe_local_indices=key_local,
            keyframe_source_indices=key_source,
            aggregation_weights=[float(value) for value in weights],
            content_tags=tags,
            text_anchor=text_anchor,
            diagnostics=diagnostics,
            text_anchor_embedding=text_anchor_embedding,
            alignment_backend=self,
        )

    def encode_script(self, text: str) -> np.ndarray:
        return self._encode_texts([text or "empty script"])[0]

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self._encode_texts(texts)

    def _resolve_device(self, requested: str) -> str:
        if requested and requested != "auto":
            if requested.startswith("cuda") and not self.torch.cuda.is_available():
                raise RuntimeError(f"Requested reward device `{requested}`, but CUDA is unavailable")
            return requested
        return "cuda:0" if self.torch.cuda.is_available() else "cpu"

    def _config_sha256(self) -> str:
        config_path = Path(self.model_path) / "config.json"
        if not config_path.exists():
            return ""
        return hashlib.sha256(config_path.read_bytes()).hexdigest()

    def _resolve_dtype(self, requested: str):
        if self.device == "cpu":
            return self.torch.float32
        mapping = {
            "float16": self.torch.float16,
            "fp16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "bf16": self.torch.bfloat16,
            "float32": self.torch.float32,
            "fp32": self.torch.float32,
        }
        if requested.lower() not in mapping:
            raise ValueError(f"Unsupported SigLIP2 dtype: {requested}")
        return mapping[requested.lower()]

    def _encode_images(self, frames: np.ndarray) -> np.ndarray:
        from PIL import Image

        images = [Image.fromarray(frame.astype(np.uint8), mode="RGB") for frame in frames]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = self._move_inputs(inputs)
        with self.torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        return self._features_to_numpy(features)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        inputs = self._move_inputs(inputs)
        with self.torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return self._features_to_numpy(features)

    def _move_inputs(self, inputs: Any) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in dict(inputs).items():
            if not hasattr(value, "to"):
                moved[key] = value
            elif key == "pixel_values" and self.device != "cpu":
                moved[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                moved[key] = value.to(self.device)
        return moved

    def _features_to_numpy(self, features: Any) -> np.ndarray:
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        elif isinstance(features, (tuple, list)):
            features = features[0]
        array = features.detach().float().cpu().numpy()
        return _l2_normalize(np.asarray(array, dtype=np.float32))

    def _tag_taxonomy(self) -> tuple[list[str], list[str], dict[str, slice]]:
        labels: list[str] = []
        prompts: list[str] = []
        slices: dict[str, slice] = {}
        for group, entries in self.TAG_GROUPS.items():
            start = len(labels)
            labels.extend(label for label, _ in entries)
            prompts.extend(prompt for _, prompt in entries)
            slices[group] = slice(start, len(labels))
        return labels, prompts, slices

    def _select_tags(self, video_embedding: np.ndarray) -> list[str]:
        similarities = self.tag_embeddings @ video_embedding
        selected: list[str] = []
        for group, group_slice in self.tag_group_slices.items():
            local_scores = similarities[group_slice]
            count = min(self.TAGS_PER_GROUP[group], len(local_scores))
            local_indices = np.argsort(local_scores)[::-1][:count]
            start = int(group_slice.start or 0)
            selected.extend(self.tag_labels[start + int(index)] for index in local_indices)
        return selected


def script_video_alignment(
    script_text: str,
    representation: VideoRepresentation,
    visual_grounding_balance: float = 0.5,
    text_anchor_semantic_balance: float = 0.7,
) -> float:
    if representation.alignment_backend is None:
        raise RuntimeError("The SigLIP2 alignment backend is unavailable.")
    if representation.text_anchor_embedding is None:
        raise RuntimeError("The SigLIP2 text-anchor embedding is unavailable.")
    script_vector = representation.alignment_backend.encode_script(script_text)
    visual_score = (cosine(script_vector, representation.video_embedding) + 1.0) / 2.0
    anchor_score = (cosine(script_vector, representation.text_anchor_embedding) + 1.0) / 2.0
    script_lower = script_text.lower()
    tag_hits = sum(
        1 for tag in representation.content_tags if tag and tag.lower() in script_lower
    )
    tag_score = tag_hits / max(1, len(representation.content_tags))
    text_evidence_score = (
        text_anchor_semantic_balance * anchor_score
        + (1.0 - text_anchor_semantic_balance) * tag_score
    )
    return float(
        visual_grounding_balance * visual_score
        + (1.0 - visual_grounding_balance) * text_evidence_score
    )


def _aggregation_weights(
    embeddings: np.ndarray,
    centrality_weight: float = 0.55,
    temperature: float = 0.35,
) -> np.ndarray:
    if len(embeddings) == 1:
        return np.ones(1, dtype=np.float32)
    center = _l2_normalize(embeddings.mean(axis=0, keepdims=True))[0]
    centrality = np.asarray([(cosine(frame, center) + 1.0) / 2.0 for frame in embeddings])
    novelty = np.zeros(len(embeddings), dtype=np.float32)
    novelty[0] = 0.5
    for idx in range(1, len(embeddings)):
        novelty[idx] = 1.0 - ((cosine(embeddings[idx], embeddings[idx - 1]) + 1.0) / 2.0)
    scores = centrality_weight * centrality + (1.0 - centrality_weight) * novelty
    scores = scores - scores.max()
    exp_scores = np.exp(scores / max(1e-8, temperature))
    return (exp_scores / exp_scores.sum()).astype(np.float32)


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(denom, 1e-8)
