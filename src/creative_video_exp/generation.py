from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import GenerationConfig
from .data import VideoSample
from .representations import VideoRepresentation


RUNNER_VARIANT = "visual_story"


@dataclass
class ScriptSegment:
    start: float | None
    end: float | None
    narration: str
    on_screen_text: str
    salient_point: str
    control_tags: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {


            "start": self.start,
            "end": self.end,
            "narration": self.narration,
            "on_screen_text": self.on_screen_text,
            "salient_point": self.salient_point,
            "control_tags": self.control_tags,
        }


@dataclass
class ScriptCandidate:
    candidate_id: str
    timeline: list[ScriptSegment]
    controls: dict[str, Any]
    text: str
    variant: str
    raw_model_variant: str = ""
    parse_issues: list[str] | None = None
    missing_required_fields: list[str] | None = None
    generation_provenance: dict[str, Any] | None = None
    schema_normalizations: list[str] | None = None
    model_authored_control_keys_ignored: list[str] | None = None
    timeline_normalization: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "candidate_id": self.candidate_id,
            "variant": self.variant,
            "controls": self.controls,
            "timeline": [segment.as_dict() for segment in self.timeline],
            "text": self.text,
        }
        if self.parse_issues is not None:
            payload["parse_issues"] = list(self.parse_issues)
        if self.missing_required_fields is not None:
            payload["missing_required_fields"] = list(self.missing_required_fields)
        if self.generation_provenance is not None:
            payload["generation_provenance"] = dict(self.generation_provenance)
        if self.timeline_normalization is not None:
            payload["timeline_normalization"] = dict(self.timeline_normalization)
        return payload


class StructuredScriptGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config

    def generate(self, sample: VideoSample, representation: VideoRepresentation) -> list[ScriptCandidate]:
        variants = [
            "content_first",
            "scene_to_value",
            "problem_solution",
            "early_summary_probe",
            "risk_probe",
            "dense_probe",
            "soft_story",
            "direct_summary",
        ]
        candidates = []
        for index in range(self.config.num_candidates):
            variant = variants[index % len(variants)]
            candidates.append(self._make_candidate(index, variant, sample, representation))
        return candidates

    def _make_candidate(
        self,
        index: int,
        variant: str,
        sample: VideoSample,
        representation: VideoRepresentation,
    ) -> ScriptCandidate:
        target_duration = float(self.config.target_duration_sec)
        segment_count = self.config.segments
        if variant == "dense_probe":
            segment_count = max(3, self.config.segments - 1)
        step = target_duration / segment_count
        points = _key_information_points(self.config)
        if variant == "problem_solution":
            points = list(reversed(points))
        visual_tag = representation.content_tags[index % len(representation.content_tags)]
        summary_segment = _summary_segment(segment_count, self.config.summary_position)
        if variant == "early_summary_probe":
            summary_segment = 1 if segment_count > 2 else 0

        timeline = []
        for segment_idx in range(segment_count):
            start = segment_idx * step
            end = (segment_idx + 1) * step
            point = points[min(segment_idx, len(points) - 1)]
            narration = self._narration(
                segment_idx=segment_idx,
                segment_count=segment_count,
                point=point,
                visual_tag=visual_tag,
                variant=variant,
                is_summary=segment_idx == summary_segment,
            )
            on_screen = self._on_screen_text(point, segment_idx == summary_segment)


            tags: list[str] = []
            if segment_idx == summary_segment:
                tags.append("summary")
            if point:
                tags.append("salient_point")
            timeline.append(
                ScriptSegment(
                    start=start,
                    end=end,
                    narration=narration,
                    on_screen_text=on_screen,
                    salient_point=point,
                    control_tags=tags,
                )
            )

        text = "\n".join(segment.narration for segment in timeline)
        controls = {
            "target_duration_sec": self.config.target_duration_sec,
            "segments": self.config.segments,
            "output_language": self.config.output_language,
            "summary_position": self.config.summary_position,
            "pace": self.config.pace,
            "information_density": self.config.information_density,
            "density_chars_per_second": self.config.density_chars_per_second(),
            "target_characters_per_segment": self.config.target_characters_per_segment(),
            "density_length_tolerance": self.config.density_length_tolerance,
            "salient_points": points,
            "risk_terms": self.config.risk_terms,
        }
        return ScriptCandidate(
            candidate_id=f"{sample.video_id}_cand_{index:02d}",
            timeline=timeline,
            controls=controls,
            text=text,
            variant=RUNNER_VARIANT,
        )

    def _narration(
        self,
        segment_idx: int,
        segment_count: int,
        point: str,
        visual_tag: str,
        variant: str,
        is_summary: bool,
    ) -> str:
        audience = "viewers"
        scene = visual_tag
        if segment_idx == 0:
            if variant == "problem_solution":
                return f"Would you like to bring the experience shown in {scene} into everyday use?"
            return f"Start with the {visual_tag}: {scene} establishes the first visual impression."
        if is_summary:
            if variant == "risk_probe":
                return f"The sequence supposedly guarantees a complete interpretation of {point} for {audience}."
            return f"The closing view summarizes {point} using only the visible evidence."
        if variant == "dense_probe":
            return f"The scene, details, and visible response highlight {point} so {audience} can assess it quickly."
        if variant == "scene_to_value":
            return f"The next view stays with {scene}, connecting the visual evidence to {point} without overclaiming."
        if variant == "soft_story":
            return f"Move gently from this detail to {point}, giving {audience} a clear and natural progression."
        return f"The next beat focuses on {point}, supported by the visible {visual_tag}."

    def _on_screen_text(self, point: str, is_summary: bool) -> str:
        if is_summary:
            return "Sequence summary"
        return point or "Key highlight"


def _key_information_points(config: GenerationConfig) -> list[str]:
    points = list(config.salient_points)
    points = [point for point in points if point]
    if not points:
        points = ["visual evidence", "temporal progression", "key-event coverage"]
    while len(points) < config.segments:
        points.append(points[-1])
    return points


def _summary_segment(segment_count: int, position: str) -> int:
    if position == "early":
        return min(1, segment_count - 1)
    if position == "middle":
        return segment_count // 2
    return max(0, segment_count - 1)
