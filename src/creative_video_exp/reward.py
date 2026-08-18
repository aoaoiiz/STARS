from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import GenerationConfig, RewardConfig
from .generation import ScriptCandidate
from .representations import VideoRepresentation, script_video_alignment
from .utils import clip01


CONTROL_CATEGORIES = (
    "segment_count",
    "timestamp_validity",
    "duration_coverage",
    "required_fields",
    "summary_position",
    "information_density",
)


@dataclass
class RewardBreakdown:
    total: float
    alignment: float
    readability: float
    rhythm: float
    control: float
    risk: float
    violations: list[str]
    risk_terms_detected: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 6),
            "alignment": round(self.alignment, 6),
            "readability": round(self.readability, 6),
            "rhythm": round(self.rhythm, 6),
            "control": round(self.control, 6),
            "risk": round(self.risk, 6),
            "control_violations": self.violations,
            "violations": self.violations,
            "risk_terms_detected": self.risk_terms_detected,
        }


class SelfRewardScorer:
    def __init__(
        self,
        reward_config: RewardConfig,
        generation_config: GenerationConfig,
    ):
        self.reward_config = reward_config
        self.generation_config = generation_config
        self.reward_config.validate_formal_protocol()

    def score(
        self,
        candidate: ScriptCandidate,
        representation: VideoRepresentation,
    ) -> RewardBreakdown:
        violations = validate_candidate(candidate, self.generation_config)
        alignment = script_video_alignment(
            _alignment_text(candidate),
            representation,
            visual_grounding_balance=self.reward_config.visual_grounding_balance,
            text_anchor_semantic_balance=self.reward_config.text_anchor_semantic_balance,
        )
        readability = _readability_score(candidate, self.generation_config)
        rhythm = _rhythm_score(candidate, self.generation_config)
        control = clip01(1.0 - len(set(violations)) / len(CONTROL_CATEGORIES))
        risk_terms_detected = _detected_risk_terms(
            _all_script_text(candidate),
            self.generation_config.risk_terms,
        )
        risk = _risk_score(risk_terms_detected, self.generation_config.risk_terms)
        total = (
            self.reward_config.alignment_weight * alignment
            + self.reward_config.readability_weight * readability
            + self.reward_config.rhythm_weight * rhythm
            + self.reward_config.control_weight * control
            + self.reward_config.risk_weight * risk
        )
        return RewardBreakdown(
            total=clip01(total),
            alignment=clip01(alignment),
            readability=clip01(readability),
            rhythm=clip01(rhythm),
            control=clip01(control),
            risk=clip01(risk),
            violations=violations,
            risk_terms_detected=risk_terms_detected,
        )


def validate_candidate(
    candidate: ScriptCandidate,
    config: GenerationConfig,
) -> list[str]:
    violations: list[str] = []
    if len(candidate.timeline) != config.segments:
        violations.append("segment_count")

    if not _timestamps_valid(candidate):
        violations.append("timestamp_validity")
    if not _duration_coverage_ok(candidate, config):
        violations.append("duration_coverage")
    if not _required_fields_present(candidate):
        violations.append("required_fields")

    summary_indices = [
        idx for idx, segment in enumerate(candidate.timeline) if "summary" in segment.control_tags
    ]
    expected = _expected_summary_index(config.segments, config.summary_position)
    if len(summary_indices) != 1 or summary_indices[0] != expected:
        violations.append("summary_position")

    density = _mean_segment_words(candidate)
    if config.information_density == "low" and density > config.target_words_per_segment:
        violations.append("information_density")
    if config.information_density == "medium" and not (
        config.min_words_per_segment <= density <= config.max_words_per_segment
    ):
        violations.append("information_density")
    if config.information_density == "high" and density < config.target_words_per_segment:
        violations.append("information_density")
    return violations


def _readability_score(candidate: ScriptCandidate, config: GenerationConfig) -> float:
    lengths = np.asarray(
        [_english_word_count(segment.narration) for segment in candidate.timeline],
        dtype=np.float32,
    )
    if len(lengths) == 0:
        return 0.0
    target = float(max(1, config.target_words_per_segment))
    closeness = 1.0 - float(np.mean(np.minimum(1.0, np.abs(lengths - target) / target)))
    within_range = float(
        np.mean(
            (lengths >= config.min_words_per_segment)
            & (lengths <= config.max_words_per_segment)
        )
    )
    sentence_completion = float(
        np.mean(
            [
                1.0 if re.search(r"[.!?][\"']?$", segment.narration.strip()) else 0.0
                for segment in candidate.timeline
            ]
        )
    )
    return clip01(0.60 * closeness + 0.25 * within_range + 0.15 * sentence_completion)


def _rhythm_score(candidate: ScriptCandidate, config: GenerationConfig) -> float:
    if not candidate.timeline or config.segments <= 0 or config.target_duration_sec <= 0:
        return 0.0
    target_duration = float(config.target_duration_sec)
    target = np.full(config.segments, target_duration / config.segments, dtype=np.float64)
    observed = np.zeros(config.segments, dtype=np.float64)
    for index, segment in enumerate(candidate.timeline[: config.segments]):
        if _finite_number(segment.start) and _finite_number(segment.end):
            observed[index] = max(0.0, float(segment.end) - float(segment.start))
    profile_error = float(np.sum(np.abs(observed - target)) / (2.0 * target_duration))
    profile_score = clip01(1.0 - profile_error)
    observed_boundaries = np.cumsum(observed)
    target_boundaries = np.cumsum(target)
    boundary_error = float(np.mean(np.abs(observed_boundaries - target_boundaries)) / target_duration)
    boundary_score = clip01(1.0 - 2.0 * boundary_error)
    extra_segment_penalty = max(0, len(candidate.timeline) - config.segments) / max(1, config.segments)
    return clip01(0.55 * profile_score + 0.45 * boundary_score - extra_segment_penalty)


def _risk_score(detected_terms: list[str], risk_terms: list[str]) -> float:
    if not risk_terms:
        return 1.0
    configured = {term.lower() for term in risk_terms if term}
    return clip01(1.0 - len(set(detected_terms)) / max(1, len(configured)))


def _expected_summary_index(segment_count: int, position: str) -> int:
    if segment_count <= 0:
        return 0
    if position == "early":
        return min(1, segment_count - 1)
    if position == "middle":
        return segment_count // 2
    return segment_count - 1


def _mean_segment_words(candidate: ScriptCandidate) -> float:
    if not candidate.timeline:
        return 0.0
    return float(np.mean([_english_word_count(segment.narration) for segment in candidate.timeline]))


def _english_word_count(text: str) -> int:
    normalized = unicodedata.normalize("NFKD", text or "")
    latin_text = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return len(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)?", latin_text))


def _finite_number(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _timestamps_valid(candidate: ScriptCandidate) -> bool:
    if not candidate.timeline:
        return False
    previous_end: float | None = None
    for segment in candidate.timeline:
        if not _finite_number(segment.start) or not _finite_number(segment.end):
            return False
        start = float(segment.start)
        end = float(segment.end)
        if start < 0.0 or end <= start:
            return False
        if previous_end is not None and start < previous_end - 1e-6:
            return False
        previous_end = end
    return True


def _duration_coverage_ok(candidate: ScriptCandidate, config: GenerationConfig) -> bool:
    if not _timestamps_valid(candidate) or config.target_duration_sec <= 0:
        return False
    target = float(config.target_duration_sec)
    first_start = float(candidate.timeline[0].start)
    last_end = float(candidate.timeline[-1].end)
    boundary_tolerance = max(0.5, 0.05 * target)
    if abs(first_start) > boundary_tolerance:
        return False
    if abs(last_end - target) / target > 0.15:
        return False
    gaps = 0.0
    for left, right in zip(candidate.timeline, candidate.timeline[1:]):
        gaps += max(0.0, float(right.start) - float(left.end))
    return gaps / target <= 0.15


def _required_fields_present(candidate: ScriptCandidate) -> bool:
    if not candidate.timeline:
        return False
    if candidate.missing_required_fields:
        return False
    return all(bool(segment.narration.strip()) for segment in candidate.timeline)


def _all_script_text(candidate: ScriptCandidate) -> str:
    fields: list[str] = [candidate.text]
    for segment in candidate.timeline:
        fields.extend(
            [
                segment.narration,
                segment.on_screen_text,
                segment.salient_point,
                " ".join(segment.control_tags),
            ]
        )
    return "\n".join(field for field in fields if field)


def _alignment_text(candidate: ScriptCandidate) -> str:
    fields: list[str] = []
    for segment in candidate.timeline:
        fields.extend(
            [
                segment.narration,
                segment.on_screen_text,
                segment.salient_point,
            ]
        )
    return "\n".join(field for field in fields if field)


def _detected_risk_terms(text: str, risk_terms: list[str]) -> list[str]:
    text_lower = text.lower()
    hits: list[str] = []
    for term in risk_terms:
        normalized = term.strip().lower()
        if normalized and normalized in text_lower and normalized not in hits:
            hits.append(normalized)
    return hits
