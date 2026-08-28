from __future__ import annotations

import math
import re
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
    "required_fields",
    "summary_position",
    "key_information_order",
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
            "total": float(self.total),
            "alignment": float(self.alignment),
            "readability": float(self.readability),
            "rhythm": float(self.rhythm),
            "control": float(self.control),
            "risk": float(self.risk),
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
        readability = _readability_score(
            candidate,
            self.generation_config,
            beta=self.reward_config.readability_balance_beta,
            sentence_bonus=self.reward_config.readability_sentence_bonus,
        )
        rhythm = _rhythm_score(
            candidate,
            self.generation_config,
            evenness_balance=self.reward_config.rhythm_evenness_balance,
            epsilon=self.reward_config.numerical_epsilon,
        )
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

    if not _timestamps_match_contract(candidate, config):
        violations.append("timestamp_validity")
    if not _required_fields_present(candidate):
        violations.append("required_fields")

    summary_indices = [
        idx for idx, segment in enumerate(candidate.timeline) if "summary" in segment.control_tags
    ]
    expected = _expected_summary_index(config.segments, config.summary_position)
    if len(summary_indices) != 1 or summary_indices[0] != expected:
        violations.append("summary_position")

    if not _key_information_order_ok(candidate, config.salient_points):
        violations.append("key_information_order")

    target_length = _density_target_length(config)
    mean_length = _mean_segment_characters(candidate)
    relative_error = abs(mean_length - target_length) / target_length
    if relative_error > max(0.0, float(config.density_length_tolerance)):
        violations.append("information_density")
    return violations


def build_preference_pairs(
    sample_id: str,
    candidate_rows: list[dict[str, Any]],
    margin: float,
) -> list[dict[str, Any]]:
    pairs = []
    sorted_rows = sorted(candidate_rows, key=lambda row: row["reward"]["total"], reverse=True)
    for left_idx, chosen in enumerate(sorted_rows):
        for rejected in sorted_rows[left_idx + 1 :]:
            diff = chosen["reward"]["total"] - rejected["reward"]["total"]
            if diff >= margin:
                pairs.append(
                    {
                        "video_id": sample_id,
                        "chosen_id": chosen["candidate_id"],
                        "rejected_id": rejected["candidate_id"],
                        "chosen_score": chosen["reward"]["total"],
                        "rejected_score": rejected["reward"]["total"],
                        "margin": round(diff, 6),
                        "chosen_text": chosen["text"],
                        "rejected_text": rejected["text"],
                    }
                )
    return pairs


def _readability_score(
    candidate: ScriptCandidate,
    config: GenerationConfig,
    *,
    beta: float,
    sentence_bonus: float,
) -> float:
    lengths = np.asarray(
        [_character_count(segment.narration) for segment in candidate.timeline],
        dtype=np.float64,
    )
    if len(lengths) == 0:
        return 0.0
    target = _density_target_length(config)
    length_penalty = float(np.mean(np.abs(lengths - target) / target))
    balance_penalty = max(0.0, float(beta)) * float(np.std(lengths)) / target
    completion_rate = float(
        np.mean(
            [
                1.0
                if re.search(
                    r"[.!?。！？][\"'”’)}\]]*$",
                    segment.narration.strip(),
                )
                else 0.0
                for segment in candidate.timeline
            ]
        )
    )
    punctuation_bonus = max(0.0, float(sentence_bonus)) * completion_rate
    return clip01(1.0 - length_penalty - balance_penalty + punctuation_bonus)


def _rhythm_score(
    candidate: ScriptCandidate,
    config: GenerationConfig,
    *,
    evenness_balance: float,
    epsilon: float,
) -> float:
    if not candidate.timeline or config.segments <= 0 or config.target_duration_sec <= 0:
        return 0.0
    target_duration = float(config.target_duration_sec)
    target = _pace_duration_profile(config)
    observed = np.zeros(config.segments, dtype=np.float64)
    for index, segment in enumerate(candidate.timeline[: config.segments]):
        if _finite_number(segment.start) and _finite_number(segment.end):
            observed[index] = max(0.0, float(segment.end) - float(segment.start))
    safe_epsilon = max(float(epsilon), 1e-12)
    uniform_duration = target_duration / config.segments
    evenness = clip01(
        1.0 - float(np.std(observed)) / (uniform_duration + safe_epsilon)
    )
    pace = clip01(
        1.0
        - float(
            np.mean(
                np.abs(observed - target) / (target + safe_epsilon)
            )
        )
    )
    balance = clip01(float(evenness_balance))
    return clip01(balance * evenness + (1.0 - balance) * pace)


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


def _mean_segment_characters(candidate: ScriptCandidate) -> float:
    if not candidate.timeline:
        return 0.0
    return float(
        np.mean([_character_count(segment.narration) for segment in candidate.timeline])
    )


def _character_count(text: str) -> int:
    return len((text or "").strip())


def _density_target_length(config: GenerationConfig) -> float:
    return max(1.0, config.target_characters_per_segment())


def _pace_duration_profile(config: GenerationConfig) -> np.ndarray:
    if config.target_duration_sec <= 0 or config.segments <= 0:
        raise ValueError("Pace profile requires positive duration and segment count.")
    if config.pace_duration_profile:
        profile = np.asarray(config.pace_duration_profile, dtype=np.float64)
        if len(profile) != config.segments or np.any(~np.isfinite(profile)) or np.any(profile <= 0):
            raise ValueError(
                "pace_duration_profile must contain one positive finite value per segment."
            )
        return profile * (float(config.target_duration_sec) / float(profile.sum()))
    if config.pace.strip().lower() not in {"medium", "uniform"}:
        raise ValueError(
            "A non-uniform pace requires an explicit pace_duration_profile."
        )
    return np.full(
        config.segments,
        float(config.target_duration_sec) / config.segments,
        dtype=np.float64,
    )


def _finite_number(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _timestamps_match_contract(
    candidate: ScriptCandidate,
    config: GenerationConfig,
) -> bool:
    if not candidate.timeline or config.target_duration_sec <= 0:
        return False
    tolerance = max(1e-6, 1e-8 * float(config.target_duration_sec))
    previous_end: float | None = None
    for index, segment in enumerate(candidate.timeline):
        if not _finite_number(segment.start) or not _finite_number(segment.end):
            return False
        start = float(segment.start)
        end = float(segment.end)
        if start < -tolerance or end <= start + tolerance:
            return False
        if index == 0 and abs(start) > tolerance:
            return False
        if previous_end is not None and abs(start - previous_end) > tolerance:
            return False
        previous_end = end
    return previous_end is not None and abs(
        previous_end - float(config.target_duration_sec)
    ) <= tolerance


def _key_information_order_ok(
    candidate: ScriptCandidate,
    expected_points: list[str],
) -> bool:
    expected = [point.strip().casefold() for point in expected_points if point.strip()]
    if not expected:
        return True
    segment_values = [segment.salient_point.casefold() for segment in candidate.timeline]
    cursor = 0
    for point in expected:
        match = next(
            (
                index
                for index in range(cursor, len(segment_values))
                if point in segment_values[index]
            ),
            None,
        )
        if match is None:
            return False
        cursor = match
    return True


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
