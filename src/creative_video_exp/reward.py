from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import GenerationConfig, RewardConfig
from .generation import ScriptCandidate
from .representations import VideoRepresentation, script_video_alignment
from .utils import clip01


@dataclass
class RewardBreakdown:
    total: float
    alignment: float
    readability: float
    rhythm: float
    control: float
    risk: float
    text_reasoning: float
    text_reasoning_rationale: str
    violations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 6),
            "alignment": round(self.alignment, 6),
            "readability": round(self.readability, 6),
            "rhythm": round(self.rhythm, 6),
            "control": round(self.control, 6),
            "risk": round(self.risk, 6),
            "text_reasoning": round(self.text_reasoning, 6),
            "text_reasoning_rationale": self.text_reasoning_rationale,
            "violations": self.violations,
        }


class SelfRewardScorer:
    def __init__(
        self,
        reward_config: RewardConfig,
        generation_config: GenerationConfig,
        text_reward_model: Any | None = None,
    ):
        self.reward_config = reward_config
        self.generation_config = generation_config
        self.text_reward_model = text_reward_model

    def score(
        self,
        candidate: ScriptCandidate,
        representation: VideoRepresentation,
    ) -> RewardBreakdown:
        violations = validate_candidate(candidate, self.generation_config)
        alignment = script_video_alignment(candidate.text, representation)
        readability = _readability_score(candidate)
        rhythm = _rhythm_score(candidate, self.generation_config)
        control = clip01(1.0 - len(violations) / 5.0)
        risk = _risk_score(candidate.text, self.generation_config.risk_terms)
        text_reasoning_result = (
            self.text_reward_model.score(candidate)
            if self.text_reward_model is not None
            else None
        )
        text_reasoning = (
            clip01(text_reasoning_result.score)
            if text_reasoning_result is not None
            else 0.0
        )
        text_reasoning_rationale = (
            text_reasoning_result.rationale
            if text_reasoning_result is not None
            else ""
        )
        weighted_total = (
            self.reward_config.alignment_weight * alignment
            + self.reward_config.readability_weight * readability
            + self.reward_config.rhythm_weight * rhythm
            + self.reward_config.control_weight * control
            + self.reward_config.risk_weight * risk
            + self.reward_config.text_reasoning_weight * text_reasoning
        )
        weight_sum = (
            self.reward_config.alignment_weight
            + self.reward_config.readability_weight
            + self.reward_config.rhythm_weight
            + self.reward_config.control_weight
            + self.reward_config.risk_weight
            + self.reward_config.text_reasoning_weight
        )
        total = weighted_total / max(1e-8, weight_sum)
        return RewardBreakdown(
            total=clip01(total),
            alignment=clip01(alignment),
            readability=clip01(readability),
            rhythm=clip01(rhythm),
            control=clip01(control),
            risk=clip01(risk),
            text_reasoning=clip01(text_reasoning),
            text_reasoning_rationale=text_reasoning_rationale,
            violations=violations,
        )


def validate_candidate(
    candidate: ScriptCandidate,
    config: GenerationConfig,
) -> list[str]:
    violations: list[str] = []
    if len(candidate.timeline) != config.segments:
        violations.append("segment_count")

    duration = candidate.timeline[-1].end - candidate.timeline[0].start if candidate.timeline else 0.0
    if config.target_duration_sec > 0:
        rel_error = abs(duration - config.target_duration_sec) / config.target_duration_sec
        if rel_error > 0.15:
            violations.append("duration")

    cta_indices = [
        idx for idx, segment in enumerate(candidate.timeline) if "cta" in segment.control_tags
    ]
    expected = _expected_cta_index(len(candidate.timeline), config.cta_position)
    if not cta_indices or abs(cta_indices[-1] - expected) > 1:
        violations.append("cta_position")

    expected_points = candidate.controls.get("selling_points") or config.selling_points
    if not _selling_order_ok(candidate, expected_points):
        violations.append("selling_point_order")

    risk_hits = [term for term in config.risk_terms if term and term in candidate.text]
    if risk_hits:
        violations.append("risk_terms:" + ",".join(risk_hits))

    density = _mean_segment_chars(candidate)
    if config.information_density == "low" and density > 34:
        violations.append("information_density")
    if config.information_density == "medium" and not (18 <= density <= 56):
        violations.append("information_density")
    if config.information_density == "high" and density < 32:
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


def _readability_score(candidate: ScriptCandidate) -> float:
    lengths = np.asarray([len(segment.narration) for segment in candidate.timeline], dtype=np.float32)
    if len(lengths) == 0:
        return 0.0
    target = 34.0
    length_score = 1.0 - float(np.mean(np.abs(lengths - target)) / target)
    variance_penalty = min(0.35, float(np.std(lengths) / max(1.0, target)))
    punctuation_bonus = 0.08 if any(mark in candidate.text for mark in "，。！？") else 0.0
    return clip01(length_score - variance_penalty + punctuation_bonus)


def _rhythm_score(candidate: ScriptCandidate, config: GenerationConfig) -> float:
    durations = np.asarray(
        [segment.end - segment.start for segment in candidate.timeline],
        dtype=np.float32,
    )
    if len(durations) == 0:
        return 0.0
    target = config.target_duration_sec / max(1, config.segments)
    evenness = 1.0 - float(np.std(durations) / max(1.0, target))
    pace_target = {"slow": 7.5, "medium": 6.0, "fast": 4.5}.get(config.pace, 6.0)
    pace_score = 1.0 - abs(float(durations.mean()) - pace_target) / max(1.0, pace_target)
    return clip01(0.55 * evenness + 0.45 * pace_score)


def _risk_score(text: str, risk_terms: list[str]) -> float:
    if not risk_terms:
        return 1.0
    hits = sum(1 for term in risk_terms if term and term in text)
    return clip01(1.0 - hits / max(1, len(risk_terms)))


def _expected_cta_index(segment_count: int, position: str) -> int:
    if segment_count <= 0:
        return 0
    if position == "early":
        return min(1, segment_count - 1)
    if position == "middle":
        return segment_count // 2
    return segment_count - 1


def _selling_order_ok(candidate: ScriptCandidate, expected_points: list[str]) -> bool:
    expected = [point for point in expected_points if point]
    if len(expected) < 2:
        return True
    text = candidate.text
    positions = []
    for point in expected[: min(3, len(expected))]:
        position = text.find(point)
        if position >= 0:
            positions.append(position)
    return len(positions) < 2 or positions == sorted(positions)


def _mean_segment_chars(candidate: ScriptCandidate) -> float:
    if not candidate.timeline:
        return 0.0
    return float(np.mean([len(segment.narration) for segment in candidate.timeline]))
