from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import cosine, hash_text_vector
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


class StatsFrameEncoder:
    """Small deterministic encoder for local experiments.

    It is intentionally lightweight. Replace this class with CLIP/SigLIP/Video-LLaVA
    when you have the real GPU budget.
    """

    def __init__(self, dim: int = 128, projection_seed: int = 20240511):
        self.dim = dim
        rng = np.random.default_rng(projection_seed)
        self.projection = rng.normal(0.0, 1.0 / np.sqrt(40), size=(40, dim)).astype(np.float32)

    def encode(self, batch: SparseFrameBatch, context_text: str = "") -> VideoRepresentation:
        stats = np.stack([_frame_stats(frame) for frame in batch.frames], axis=0)
        embeddings = stats @ self.projection
        embeddings = _l2_normalize(embeddings)
        weights = _aggregation_weights(embeddings)
        video_embedding = _l2_normalize((embeddings * weights[:, None]).sum(axis=0, keepdims=True))[0]
        key_local = np.argsort(weights)[::-1][: min(4, len(weights))].tolist()
        key_local = sorted(int(idx) for idx in key_local)
        key_source = [batch.selected_indices[idx] for idx in key_local]
        tags = _content_tags(batch.frames)
        text_anchor = " ".join(tags + [context_text])
        diagnostics = {
            "source_kind": batch.source_kind,
            "mean_brightness": float(np.mean(stats[:, 0])),
            "mean_saturation": float(np.mean(stats[:, 1])),
            "mean_motion": float(_motion_score(batch.frames)),
        }
        return VideoRepresentation(
            frame_embeddings=embeddings,
            video_embedding=video_embedding,
            keyframe_local_indices=key_local,
            keyframe_source_indices=key_source,
            aggregation_weights=[float(value) for value in weights],
            content_tags=tags,
            text_anchor=text_anchor,
            diagnostics=diagnostics,
        )


def script_video_alignment(script_text: str, representation: VideoRepresentation) -> float:
    script_vector = hash_text_vector(script_text, dim=len(representation.video_embedding))
    visual_text_vector = hash_text_vector(representation.text_anchor, dim=len(representation.video_embedding))
    text_alignment = cosine(script_vector, visual_text_vector)
    tag_hits = sum(1 for tag in representation.content_tags if tag and tag in script_text)
    tag_score = tag_hits / max(1, len(representation.content_tags))
    return float(0.7 * ((text_alignment + 1.0) / 2.0) + 0.3 * tag_score)


def _frame_stats(frame: np.ndarray) -> np.ndarray:
    arr = frame.astype(np.float32) / 255.0
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    brightness = float(arr.mean())
    saturation = float((maxc - minc).mean())
    contrast = float(arr.std())
    warmth = float(mean[0] - mean[2])
    edge_x = np.abs(np.diff(arr, axis=1)).mean()
    edge_y = np.abs(np.diff(arr, axis=0)).mean()
    hist_features = []
    for channel in range(3):
        hist, _ = np.histogram(arr[..., channel], bins=8, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        hist = hist / max(1.0, hist.sum())
        hist_features.extend(hist.tolist())
    features = [
        brightness,
        saturation,
        contrast,
        warmth,
        float(edge_x),
        float(edge_y),
        *mean.tolist(),
        *std.tolist(),
        *hist_features,
    ]
    padded = np.zeros(40, dtype=np.float32)
    padded[: min(len(features), 40)] = np.asarray(features[:40], dtype=np.float32)
    return padded


def _aggregation_weights(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) == 1:
        return np.ones(1, dtype=np.float32)
    center = _l2_normalize(embeddings.mean(axis=0, keepdims=True))[0]
    centrality = np.asarray([(cosine(frame, center) + 1.0) / 2.0 for frame in embeddings])
    novelty = np.zeros(len(embeddings), dtype=np.float32)
    novelty[0] = 0.5
    for idx in range(1, len(embeddings)):
        novelty[idx] = 1.0 - ((cosine(embeddings[idx], embeddings[idx - 1]) + 1.0) / 2.0)
    scores = 0.55 * centrality + 0.45 * novelty
    scores = scores - scores.max()
    exp_scores = np.exp(scores / 0.35)
    return (exp_scores / exp_scores.sum()).astype(np.float32)


def _content_tags(frames: np.ndarray) -> list[str]:
    arr = frames.astype(np.float32) / 255.0
    mean_rgb = arr.mean(axis=(0, 1, 2))
    brightness = float(arr.mean())
    saturation = float((arr.max(axis=3) - arr.min(axis=3)).mean())
    motion = _motion_score(frames)
    color = _dominant_color(mean_rgb)
    light = "明亮画面" if brightness >= 0.58 else "柔和光线" if brightness >= 0.38 else "低照度氛围"
    texture = "细节丰富" if saturation >= 0.22 else "色彩克制"
    tempo = "动态切换" if motion >= 0.06 else "稳定展示"
    return [color, light, texture, tempo]


def _dominant_color(mean_rgb: np.ndarray) -> str:
    red, green, blue = mean_rgb.tolist()
    if red - blue > 0.08 and red - green > 0.02:
        return "暖色调"
    if blue - red > 0.08:
        return "冷色调"
    if green > red and green > blue:
        return "自然绿色"
    return "均衡色彩"


def _motion_score(frames: np.ndarray) -> float:
    if len(frames) < 2:
        return 0.0
    arr = frames.astype(np.float32) / 255.0
    return float(np.abs(np.diff(arr, axis=0)).mean())


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(denom, 1e-8)
