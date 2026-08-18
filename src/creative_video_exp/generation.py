from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
        return payload
