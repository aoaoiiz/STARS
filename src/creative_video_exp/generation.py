from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import GenerationConfig
from .data import VideoSample
from .representations import VideoRepresentation


@dataclass
class ScriptSegment:
    start: float
    end: float
    narration: str
    on_screen_text: str
    selling_point: str
    control_tags: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "narration": self.narration,
            "on_screen_text": self.on_screen_text,
            "selling_point": self.selling_point,
            "control_tags": self.control_tags,
        }


@dataclass
class ScriptCandidate:
    candidate_id: str
    timeline: list[ScriptSegment]
    controls: dict[str, Any]
    text: str
    variant: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "variant": self.variant,
            "controls": self.controls,
            "timeline": [segment.as_dict() for segment in self.timeline],
            "text": self.text,
        }


class StructuredScriptGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config

    def generate(self, sample: VideoSample, representation: VideoRepresentation) -> list[ScriptCandidate]:
        variants = [
            "content_first",
            "scene_to_value",
            "problem_solution",
            "early_cta_probe",
            "risk_probe",
            "dense_probe",
            "soft_story",
            "direct_offer",
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
        duration = float(sample.duration or self.config.target_duration_sec)
        target_duration = float(self.config.target_duration_sec or duration)
        segment_count = self.config.segments
        if variant == "dense_probe":
            segment_count = max(3, self.config.segments - 1)
        step = target_duration / segment_count
        points = _selling_points(sample, self.config)
        if variant == "problem_solution":
            points = list(reversed(points))
        visual_tag = representation.content_tags[index % len(representation.content_tags)]
        cta_segment = _cta_segment(segment_count, self.config.cta_position)
        if variant == "early_cta_probe":
            cta_segment = 1 if segment_count > 2 else 0

        timeline = []
        for segment_idx in range(segment_count):
            start = segment_idx * step
            end = (segment_idx + 1) * step
            point = points[min(segment_idx, len(points) - 1)]
            narration = self._narration(
                segment_idx=segment_idx,
                segment_count=segment_count,
                point=point,
                sample=sample,
                visual_tag=visual_tag,
                variant=variant,
                is_cta=segment_idx == cta_segment,
            )
            on_screen = self._on_screen_text(point, sample, segment_idx == cta_segment)
            tags = ["hook"] if segment_idx == 0 else []
            if segment_idx == cta_segment:
                tags.append("cta")
            if point:
                tags.append("selling_point")
            timeline.append(
                ScriptSegment(
                    start=start,
                    end=end,
                    narration=narration,
                    on_screen_text=on_screen,
                    selling_point=point,
                    control_tags=tags,
                )
            )

        text = "\n".join(segment.narration for segment in timeline)
        controls = {
            "target_duration_sec": self.config.target_duration_sec,
            "segments": self.config.segments,
            "cta_position": self.config.cta_position,
            "pace": self.config.pace,
            "information_density": self.config.information_density,
            "selling_points": points,
            "risk_terms": self.config.risk_terms,
        }
        return ScriptCandidate(
            candidate_id=f"{sample.video_id}_cand_{index:02d}",
            timeline=timeline,
            controls=controls,
            text=text,
            variant=variant,
        )

    def _narration(
        self,
        segment_idx: int,
        segment_count: int,
        point: str,
        sample: VideoSample,
        visual_tag: str,
        variant: str,
        is_cta: bool,
    ) -> str:
        audience = sample.target_audience or "正在比较选择的人"
        cta = sample.cta or "点击了解更多"
        scene = _scene_phrase(sample.caption, visual_tag)
        if segment_idx == 0:
            if variant == "problem_solution":
                return f"是不是也想把{scene}里的体验，直接搬进日常使用？"
            return f"先看这个{visual_tag}的画面，{scene}已经把第一眼感受讲清楚。"
        if segment_idx == 1 and sample.answer:
            return f"结合题目线索，画面里的关键信息指向「{sample.answer}」，这一点支撑{point}。"
        if is_cta:
            if variant == "risk_probe":
                return f"{point}做到绝对领先，适合{audience}，现在{cta}。"
            return f"如果你也在意{point}，可以现在{cta}，把选择留给真实需求。"
        if variant == "dense_probe":
            return f"{point}、场景、细节和反馈集中出现，让{audience}快速判断它是否值得尝试。"
        if variant == "scene_to_value":
            return f"镜头继续落在{scene}，对应的价值是{point}，信息不用多但要准确。"
        if variant == "soft_story":
            return f"从这个细节过渡到{point}，语气放轻，让{audience}自然跟上节奏。"
        return f"第二层信息聚焦{point}，用画面里的{visual_tag}来支撑这个卖点。"

    def _on_screen_text(self, point: str, sample: VideoSample, is_cta: bool) -> str:
        if is_cta:
            return sample.cta or "了解更多"
        return point or "核心亮点"


def _selling_points(sample: VideoSample, config: GenerationConfig) -> list[str]:
    points = sample.selling_points or config.selling_points
    points = [point for point in points if point]
    if not points:
        points = ["核心卖点", "使用场景", "信任背书"]
    while len(points) < config.segments:
        points.append(points[-1])
    return points


def _cta_segment(segment_count: int, position: str) -> int:
    if position == "early":
        return min(1, segment_count - 1)
    if position == "middle":
        return segment_count // 2
    return max(0, segment_count - 1)


def _scene_phrase(caption: str, visual_tag: str) -> str:
    if "咖啡" in caption:
        return "咖啡滴落和桌面细节"
    if "跑步" in caption or "鞋" in caption:
        return "路面切换和鞋底纹理"
    if "灯" in caption:
        return "灯光亮度和书桌场景"
    if caption:
        return caption[:18]
    return visual_tag
