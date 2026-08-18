from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import EvaluationConfig
from .generation import ScriptCandidate
from .utils import clip01, normalize_text


FORMAL_REFERENCE_PROTOCOL = "official_qa_evidence_and_answer_v2"
FORMAL_REFERENCE_PREFIXES = ("Evidence context: ", "Expected answer: ")


def formal_reference_contract_issue(row: dict[str, Any]) -> str:
    points = row.get("semantic_points")
    if not isinstance(points, list) or len(points) != len(FORMAL_REFERENCE_PREFIXES):
        return "semantic_points must contain exactly two strings"
    normalized = [normalize_text(point) for point in points]
    if any(not point for point in normalized):
        return "semantic_points contains an empty point"
    for point, prefix in zip(normalized, FORMAL_REFERENCE_PREFIXES):
        if not point.startswith(prefix) or not point[len(prefix) :].strip():
            return f"semantic point does not follow `{prefix}...`"

    provenance = row.get("semantic_point_provenance")
    if not isinstance(provenance, dict):
        return "semantic_point_provenance is missing"
    if provenance.get("protocol") != FORMAL_REFERENCE_PROTOCOL:
        return "per-row reference protocol is invalid"
    for field in (
        "reference_enters_generation",
        "reference_enters_reward",
        "reference_enters_candidate_selection",
    ):
        if provenance.get(field) is not False:
            return f"{field} must be false"
    return ""


@dataclass(frozen=True)
class SemanticPointReference:
    points: tuple[str, ...]
    embeddings: np.ndarray | None
    source_field: str
    encoder: str
    similarity_threshold: float
    text_fields: tuple[str, ...]

    @property
    def available(self) -> bool:
        return bool(self.points) and self.embeddings is not None

    def audit_dict(self) -> dict[str, Any]:
        return {
            "reference_available": self.available,
            "reference_count": len(self.points),
            "reference_source_field": self.source_field,
            "encoder": self.encoder,
            "similarity": "cosine",
            "similarity_threshold": self.similarity_threshold,
            "script_text_fields": list(self.text_fields),
            "reference_usage": "evaluation_only",
        }


class SemanticPointEvaluator:
    def __init__(self, text_encoder: Any, config: EvaluationConfig):
        config.validate()
        self.text_encoder = text_encoder
        self.config = config
        if config.semantic_point_coverage_enabled and not hasattr(
            text_encoder, "encode_texts"
        ):
            raise TypeError(
                "SPC requires an encoder with encode_texts(list[str]); use the "
                "active frozen reward encoder."
            )

    def prepare(
        self,
        points: list[str],
        source_field: str = "",
    ) -> SemanticPointReference:
        normalized = tuple(
            dict.fromkeys(
                point
                for point in (normalize_text(value) for value in points)
                if point
            )
        )
        embeddings = None
        if self.config.semantic_point_coverage_enabled and normalized:
            embeddings = np.asarray(
                self.text_encoder.encode_texts(list(normalized)),
                dtype=np.float32,
            )
            if embeddings.ndim != 2 or embeddings.shape[0] != len(normalized):
                raise RuntimeError(
                    "SPC text encoder returned an invalid reference embedding matrix."
                )
        return SemanticPointReference(
            points=normalized,
            embeddings=embeddings,
            source_field=source_field,
            encoder=self.config.semantic_point_encoder,
            similarity_threshold=float(
                self.config.semantic_point_similarity_threshold
            ),
            text_fields=tuple(self.config.semantic_point_text_fields),
        )

    def evaluate(
        self,
        candidate: ScriptCandidate,
        reference: SemanticPointReference,
    ) -> dict[str, Any]:
        base = {
            "semantic_point_coverage": None,
            "semantic_point_reference_available": float(reference.available),
            "semantic_point_reference_count": len(reference.points),
            "semantic_point_covered_count": 0,
            "semantic_point_similarity_threshold": reference.similarity_threshold,
            "semantic_point_best_similarities": [],
            "semantic_point_best_segment_indices": [],
        }
        if not reference.available:
            return base

        indexed_segment_texts = [
            (
                index,
                self._segment_text(segment, reference.text_fields),
            )
            for index, segment in enumerate(candidate.timeline, start=1)
        ]
        indexed_segment_texts = [
            (index, text)
            for index, text in indexed_segment_texts
            if text
        ]
        segment_indices = [index for index, _ in indexed_segment_texts]
        segment_texts = [text for _, text in indexed_segment_texts]
        if not segment_texts:
            segment_texts = [normalize_text(candidate.text)] if candidate.text else []
            segment_indices = [0] if segment_texts else []
        if not segment_texts:
            return {
                **base,
                "semantic_point_coverage": 0.0,
            }

        segment_embeddings = np.asarray(
            self.text_encoder.encode_texts(segment_texts),
            dtype=np.float32,
        )
        if segment_embeddings.ndim != 2:
            raise RuntimeError(
                "SPC text encoder returned an invalid script embedding matrix."
            )
        similarities = np.clip(
            np.asarray(reference.embeddings) @ segment_embeddings.T,
            -1.0,
            1.0,
        )
        best_indices = np.argmax(similarities, axis=1)
        best_scores = similarities[
            np.arange(similarities.shape[0]),
            best_indices,
        ]
        covered = best_scores >= reference.similarity_threshold
        coverage = clip01(float(np.mean(covered)))
        return {
            **base,
            "semantic_point_coverage": round(coverage, 6),
            "semantic_point_covered_count": int(np.sum(covered)),
            "semantic_point_best_similarities": [
                round(float(score), 6) for score in best_scores
            ],
            "semantic_point_best_segment_indices": [
                segment_indices[int(index)] for index in best_indices
            ],
        }

    @staticmethod
    def _segment_text(segment: Any, fields: tuple[str, ...]) -> str:
        return " ".join(
            value
            for value in (
                normalize_text(getattr(segment, field, "")) for field in fields
            )
            if value
        )
