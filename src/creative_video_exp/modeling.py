from __future__ import annotations

import json
import base64
import io
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from .config import GenerationConfig, ModelEndpointConfig, ModelSuiteConfig
from .data import VideoSample
from .generation import ScriptCandidate, ScriptSegment
from .representations import StatsFrameEncoder
from .utils import clip01, normalize_text
from .video import SparseFrameBatch


class TextRewardModel(Protocol):
    model_report: dict[str, Any]

    def score(self, candidate: ScriptCandidate) -> "TextRewardResult":
        ...


class ScriptGenerationModel(Protocol):
    model_report: dict[str, Any]

    def generate(
        self,
        sample: VideoSample,
        batch: SparseFrameBatch,
        config: GenerationConfig,
    ) -> list[ScriptCandidate]:
        ...


@dataclass
class TextRewardResult:
    score: float
    rationale: str = ""
    model_id: str = ""
    provider: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(clip01(self.score), 6),
            "rationale": self.rationale,
            "model_id": self.model_id,
            "provider": self.provider,
        }


class NoOpTextRewardModel:
    def __init__(self, reason: str = "disabled"):
        self.model_report = {"runtime": "noop", "reason": reason}

    def score(self, candidate: ScriptCandidate) -> TextRewardResult:
        return TextRewardResult(score=0.0, rationale="text-only reward disabled")


class NoOpScriptGenerationModel:
    def __init__(self, reason: str = "disabled"):
        self.model_report = {"runtime": "rule_generator", "reason": reason}

    def generate(
        self,
        sample: VideoSample,
        batch: SparseFrameBatch,
        config: GenerationConfig,
    ) -> list[ScriptCandidate]:
        return []


class OpenAICompatibleScriptGenerationModel:
    """Call a server-hosted video-language model through a chat-completions API."""

    def __init__(self, endpoint: ModelEndpointConfig):
        self.endpoint = endpoint
        self.model_report = {
            "runtime": "openai_compatible_multimodal",
            "model_id": endpoint.id,
            "model_name": endpoint.name,
            "provider": endpoint.provider,
            "endpoint_url": endpoint.endpoint_url,
            "max_frames": endpoint.max_frames,
            "adapter": endpoint.adapter,
        }

    def generate(
        self,
        sample: VideoSample,
        batch: SparseFrameBatch,
        config: GenerationConfig,
    ) -> list[ScriptCandidate]:
        if not self.endpoint.endpoint_url:
            return []
        candidates: list[ScriptCandidate] = []
        seen_ids: set[str] = set()
        attempts = max(1, min(config.num_candidates, int(self.endpoint.retry_count or 0) + 2))
        parse_failures = 0
        last_excerpt = ""
        for attempt in range(attempts):
            missing = config.num_candidates - len(candidates)
            if missing <= 0:
                break
            prompt = _script_prompt(sample, batch, config)
            if attempt:
                prompt += (
                    " Previous responses produced too few valid JSON candidates. "
                    f"Return ONLY a JSON array with exactly {missing} NEW candidates. "
                    f"Do not reuse these candidate_id values: {sorted(seen_ids)}. "
                )
            payload = {
                "model": self.endpoint.name,
                "temperature": self.endpoint.temperature,
                "max_tokens": self.endpoint.max_new_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *_frame_content(batch, self.endpoint.max_frames),
                        ],
                    }
                ],
            }
            try:
                response_payload = _post_chat_completion(self.endpoint, payload)
            except RuntimeError as exc:
                self.model_report["last_error"] = str(exc)
                continue
            content = _message_content(response_payload)
            last_excerpt = content[:500]
            parsed = _parse_script_candidates(content, sample, config, self.endpoint)
            if not parsed:
                parse_failures += 1
                wrapped = _candidate_from_text_response(
                    content,
                    sample,
                    config,
                    self.endpoint,
                    attempt,
                )
                if wrapped is None:
                    continue
                parsed = [wrapped]
            for candidate in parsed:
                if candidate.candidate_id in seen_ids:
                    candidate.candidate_id = _candidate_id(
                        f"vlm_retry_{attempt}_{len(candidates):02d}",
                        sample.video_id,
                        len(candidates),
                    )
                seen_ids.add(candidate.candidate_id)
                candidates.append(candidate)
                if len(candidates) >= config.num_candidates:
                    break
        self.model_report["generation_attempts"] = min(attempts, max(1, len(candidates)))
        self.model_report["parsed_candidates"] = len(candidates)
        if parse_failures:
            self.model_report["parse_failures"] = parse_failures
        if len(candidates) < config.num_candidates:
            self.model_report["last_parse_error"] = (
                f"parsed {len(candidates)} of {config.num_candidates} requested candidates"
            )
            self.model_report["last_response_excerpt"] = last_excerpt
        return candidates[: config.num_candidates]


class OpenAICompatibleTextRewardModel:
    """Call a text-only reasoning/reward model through a chat-completions API."""

    def __init__(self, endpoint: ModelEndpointConfig):
        self.endpoint = endpoint
        self.model_report = {
            "runtime": "openai_compatible_text_reward",
            "model_id": endpoint.id,
            "model_name": endpoint.name,
            "provider": endpoint.provider,
            "endpoint_url": endpoint.endpoint_url,
            "adapter": endpoint.adapter,
        }

    def score(self, candidate: ScriptCandidate) -> TextRewardResult:
        if not self.endpoint.endpoint_url:
            return TextRewardResult(
                score=0.0,
                rationale="missing text reward endpoint_url",
                model_id=self.endpoint.id,
                provider=self.endpoint.provider,
            )
        payload = {
            "model": self.endpoint.name,
            "temperature": self.endpoint.temperature,
            "max_tokens": self.endpoint.max_new_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": _text_reward_prompt(candidate),
                }
            ],
        }
        try:
            response_payload = _post_chat_completion(self.endpoint, payload)
        except RuntimeError as exc:
            self.model_report["last_error"] = str(exc)
            return TextRewardResult(
                score=0.0,
                rationale=f"text reward endpoint error: {exc}",
                model_id=self.endpoint.id,
                provider=self.endpoint.provider,
            )
        content = _message_content(response_payload)
        payload = _extract_json_object(content) or {}
        score = payload.get("score", payload.get("reward", 0.0))
        rationale = normalize_text(payload.get("rationale", payload.get("reason", content[:400])))
        return TextRewardResult(
            score=clip01(_safe_float(score, 0.0)),
            rationale=rationale,
            model_id=self.endpoint.id,
            provider=self.endpoint.provider,
        )


class HeuristicTextRewardModel:
    """Local text-only stand-in for debugging reward plumbing."""

    def __init__(self, endpoint: ModelEndpointConfig):
        self.endpoint = endpoint
        self.model_report = {
            "runtime": "heuristic_text_reward",
            "model_id": endpoint.id,
            "model_name": endpoint.name,
            "provider": endpoint.provider,
        }

    def score(self, candidate: ScriptCandidate) -> TextRewardResult:
        text = candidate.text
        segment_lengths = [len(segment.narration) for segment in candidate.timeline]
        has_cta = any("cta" in segment.control_tags for segment in candidate.timeline)
        has_structure = len(candidate.timeline) >= 3 and all(segment.narration for segment in candidate.timeline)
        avg_len = sum(segment_lengths) / max(1, len(segment_lengths))
        length_score = 1.0 - min(1.0, abs(avg_len - 38.0) / 38.0)
        clarity = 0.2 if any(mark in text for mark in "，。！？") else 0.0
        score = 0.45 * length_score + 0.35 * float(has_structure) + 0.2 * float(has_cta) + clarity
        return TextRewardResult(
            score=clip01(score),
            rationale="local heuristic text-only reward for smoke/debug runs",
            model_id=self.endpoint.id,
            provider=self.endpoint.provider,
        )


class CachedTextRewardModel:
    """Read text-only reward scores precomputed by DeepSeek-R1/Llama/etc."""

    def __init__(self, endpoint: ModelEndpointConfig):
        self.endpoint = endpoint
        self.rows = _load_reward_cache(endpoint.cache_path)
        self.model_report = {
            "runtime": "external_jsonl_cache",
            "model_id": endpoint.id,
            "model_name": endpoint.name,
            "provider": endpoint.provider,
            "cache_path": endpoint.cache_path,
            "cached_rows": len(self.rows),
        }

    def score(self, candidate: ScriptCandidate) -> TextRewardResult:
        row = self.rows.get(candidate.candidate_id)
        if not row:
            return TextRewardResult(
                score=0.0,
                rationale="missing cached text-only reward",
                model_id=self.endpoint.id,
                provider=self.endpoint.provider,
            )
        return TextRewardResult(
            score=clip01(float(row.get("score", row.get("reward", 0.0)))),
            rationale=normalize_text(row.get("rationale", "")),
            model_id=self.endpoint.id,
            provider=self.endpoint.provider,
        )


def build_frame_encoder(model_suite: ModelSuiteConfig) -> tuple[StatsFrameEncoder, dict[str, Any]]:
    endpoint = model_suite.get(model_suite.active_video_model)
    if endpoint is None:
        return StatsFrameEncoder(dim=128), {
            "runtime": "fallback_stats",
            "reason": f"active_video_model `{model_suite.active_video_model}` not found",
        }

    if endpoint.adapter == "stats_frame_encoder" or endpoint.provider == "local":
        return StatsFrameEncoder(dim=128), {
            "runtime": "local_stats",
            "model_id": endpoint.id,
            "model_name": endpoint.name,
            "provider": endpoint.provider,
        }

    if endpoint.provider == "openai_compatible":
        return StatsFrameEncoder(dim=128), {
            "runtime": "auxiliary_stats_for_reward",
            "model_id": endpoint.id,
            "model_name": endpoint.name,
            "provider": endpoint.provider,
            "reason": "script generation is handled by the server VLM; local stats encoder is kept for lightweight diagnostics and reward features",
        }

    if not model_suite.allow_heavy_model_load:
        return StatsFrameEncoder(dim=128), {
            "runtime": "fallback_stats",
            "requested_model_id": endpoint.id,
            "requested_model_name": endpoint.name,
            "requested_adapter": endpoint.adapter,
            "reason": "allow_heavy_model_load=false; using local StatsFrameEncoder",
        }

    raise NotImplementedError(
        "Heavy multimodal adapters are configured but not loaded in this local runner. "
        "Use the model config to run on a GPU server, or precompute outputs and feed them back "
        "through an external_jsonl cache adapter."
    )


def build_text_reward_model(model_suite: ModelSuiteConfig) -> TextRewardModel:
    if not model_suite.active_text_reward_model:
        return NoOpTextRewardModel()
    endpoint = model_suite.get(model_suite.active_text_reward_model)
    if endpoint is None:
        return NoOpTextRewardModel(
            reason=f"active_text_reward_model `{model_suite.active_text_reward_model}` not found"
        )
    if not endpoint.enabled:
        return NoOpTextRewardModel(reason=f"{endpoint.id} disabled")
    if endpoint.provider == "heuristic" or endpoint.adapter == "heuristic_text_reward":
        return HeuristicTextRewardModel(endpoint)
    if endpoint.provider == "external_jsonl" or endpoint.adapter == "cached_text_reward":
        return CachedTextRewardModel(endpoint)
    if endpoint.provider in {"openai_compatible", "openai_compatible_text"}:
        if not model_suite.allow_heavy_model_load:
            return NoOpTextRewardModel(
                reason=(
                    f"{endpoint.name} configured as {endpoint.provider}, but "
                    "allow_heavy_model_load=false"
                )
            )
        return OpenAICompatibleTextRewardModel(endpoint)
    if not model_suite.allow_heavy_model_load:
        return NoOpTextRewardModel(
            reason=(
                f"{endpoint.name} configured as {endpoint.provider}, but "
                "allow_heavy_model_load=false"
            )
        )
    raise NotImplementedError(
        "Online/local text LLM reward adapters should be implemented in your GPU/API runtime. "
        "For reproducible paper experiments, export their scores to JSONL and use external_jsonl."
    )


def build_script_generation_model(model_suite: ModelSuiteConfig) -> ScriptGenerationModel:
    endpoint = model_suite.get(model_suite.active_video_model)
    if endpoint is None:
        return NoOpScriptGenerationModel(
            reason=f"active_video_model `{model_suite.active_video_model}` not found"
        )
    if not endpoint.enabled:
        return NoOpScriptGenerationModel(reason=f"{endpoint.id} disabled")
    if endpoint.provider == "openai_compatible":
        if not model_suite.allow_heavy_model_load:
            return NoOpScriptGenerationModel(
                reason=f"{endpoint.id} configured, but allow_heavy_model_load=false"
            )
        return OpenAICompatibleScriptGenerationModel(endpoint)
    return NoOpScriptGenerationModel(
        reason=(
            f"{endpoint.id} uses local/rule fallback in this runner; "
            "set provider=openai_compatible on a GPU server for true VLM generation"
        )
    )


def _load_reward_cache(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    cache_path = Path(os.path.expandvars(path)).expanduser()
    if not cache_path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            candidate_id = normalize_text(payload.get("candidate_id", ""))
            if candidate_id:
                rows[candidate_id] = payload
    return rows


def _script_prompt(sample: VideoSample, batch: SparseFrameBatch, config: GenerationConfig) -> str:
    frame_trace = [
        {"frame": idx, "source_index": source_idx, "bin": bin_id}
        for idx, (source_idx, bin_id) in enumerate(zip(batch.selected_indices, batch.bin_ids))
    ]
    return (
        "You are a multimodal short-video creative script generator. "
        "Generate structured Chinese ad scripts grounded in the provided video frames. "
        "The attached images are sparse bin-wise samples from the source video; use visible evidence "
        "from these frames instead of inventing unrelated product claims. "
        "Return ONLY a valid JSON array, with no markdown, no explanation, and no surrounding text. "
        "Each item must contain candidate_id, variant, controls, and timeline. "
        "Each timeline must contain exactly the requested number of segments. "
        "Each timeline item must contain numeric start/end plus narration, on_screen_text, selling_point, control_tags. "
        "Use the exact control tag `cta` for the CTA segment and `selling_point` when a segment carries a selling point. "
        f"Generate {config.num_candidates} candidates; target duration {config.target_duration_sec}s; "
        f"{config.segments} timeline segments; CTA position {config.cta_position}; "
        f"pace {config.pace}; information density {config.information_density}. "
        f"Selling point order to preserve: {config.selling_points}. "
        f"Risk terms to avoid: {config.risk_terms}. "
        f"Sparse sampling trace: source_kind={batch.source_kind}; frames={frame_trace}. "
        f"Dataset context: question={sample.question}; answer={sample.answer}; "
        f"caption={sample.caption}; category={sample.category}; "
        f"selling_points={sample.selling_points}; audience={sample.target_audience}; cta={sample.cta}."
    )


def _text_reward_prompt(candidate: ScriptCandidate) -> str:
    payload = {
        "candidate_id": candidate.candidate_id,
        "variant": candidate.variant,
        "controls": candidate.controls,
        "timeline": [segment.as_dict() for segment in candidate.timeline],
        "text": candidate.text,
    }
    return (
        "You are a text-only reward/reasoning judge for structured Chinese short-video ad scripts. "
        "Do not infer unseen visual facts. Score only the provided script text for clarity, CTA timing, "
        "selling point order, pacing comfort, risk wording, and internal consistency. "
        "Return ONLY a JSON object with fields score (0 to 1) and rationale. "
        f"Script payload: {json.dumps(payload, ensure_ascii=False)}"
    )


def _request_headers(endpoint: ModelEndpointConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key_env:
        api_key = os.environ.get(endpoint.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_chat_completion(endpoint: ModelEndpointConfig, payload: dict[str, Any]) -> dict[str, Any]:
    attempts = max(1, int(endpoint.retry_count or 0) + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            endpoint.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=_request_headers(endpoint),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=endpoint.request_timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise RuntimeError(str(last_error) if last_error else "unknown endpoint error")


def _message_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(normalize_text(item.get("text", item.get("content", ""))))
            else:
                parts.append(normalize_text(item))
        return "\n".join(part for part in parts if part)
    return normalize_text(content)


def _frame_content(batch: SparseFrameBatch, max_frames: int) -> list[dict[str, Any]]:
    frames = batch.frames[: max(1, max_frames)]
    content = []
    for frame in frames:
        image = Image.fromarray(frame).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
    return content


def _parse_script_candidates(
    content: str,
    sample: VideoSample,
    config: GenerationConfig,
    endpoint: ModelEndpointConfig | None = None,
) -> list[ScriptCandidate]:
    payload = _extract_json(content)
    if not isinstance(payload, list):
        return []
    candidates: list[ScriptCandidate] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        timeline = []
        for segment in item.get("timeline", []):
            if not isinstance(segment, dict):
                continue
            timeline.append(
                ScriptSegment(
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    narration=normalize_text(segment.get("narration", "")),
                    on_screen_text=normalize_text(segment.get("on_screen_text", "")),
                    selling_point=normalize_text(segment.get("selling_point", "")),
                    control_tags=_normalize_control_tags(segment.get("control_tags", [])),
                )
            )
        if not timeline:
            continue
        candidate_id = _candidate_id(item.get("candidate_id", ""), sample.video_id, idx)
        text = "\n".join(segment.narration for segment in timeline)
        controls = item.get("controls") if isinstance(item.get("controls"), dict) else {}
        candidate = ScriptCandidate(
                candidate_id=candidate_id,
                timeline=timeline,
                controls={
                    "target_duration_sec": config.target_duration_sec,
                    "segments": config.segments,
                    "cta_position": config.cta_position,
                    "pace": config.pace,
                    "information_density": config.information_density,
                    "selling_points": controls.get(
                        "selling_points",
                        sample.selling_points or config.selling_points,
                    ),
                    "risk_terms": config.risk_terms,
                    "generated_by": "openai_compatible_multimodal",
                    "model_id": endpoint.id if endpoint else "",
                    "model_name": endpoint.name if endpoint else "",
                },
                text=text,
                variant=normalize_text(item.get("variant", "vlm")),
            )
        candidates.append(_normalize_controlled_candidate(candidate, sample, config))
    return candidates[: config.num_candidates]


def _candidate_from_text_response(
    content: str,
    sample: VideoSample,
    config: GenerationConfig,
    endpoint: ModelEndpointConfig | None,
    attempt: int,
) -> ScriptCandidate | None:
    text = _clean_model_text(content)
    if not text:
        return None
    points = [point for point in (sample.selling_points or config.selling_points) if point]
    chunks = _split_text_segments(text, max(1, config.segments))
    step = float(config.target_duration_sec or 30) / max(1, config.segments)
    timeline = []
    for idx in range(max(1, config.segments)):
        point = points[min(idx, len(points) - 1)] if points else ""
        narration = chunks[idx] if idx < len(chunks) else ""
        timeline.append(
            ScriptSegment(
                start=idx * step,
                end=(idx + 1) * step,
                narration=narration,
                on_screen_text=point or "核心亮点",
                selling_point=point,
                control_tags=["selling_point"],
            )
        )
    candidate = ScriptCandidate(
        candidate_id=_candidate_id(f"vlm_wrapped_{attempt:02d}", sample.video_id, attempt),
        timeline=timeline,
        controls={
            "target_duration_sec": config.target_duration_sec,
            "segments": config.segments,
            "cta_position": config.cta_position,
            "pace": config.pace,
            "information_density": config.information_density,
            "selling_points": points or config.selling_points,
            "risk_terms": config.risk_terms,
            "generated_by": "openai_compatible_multimodal",
            "model_id": endpoint.id if endpoint else "",
            "model_name": endpoint.name if endpoint else "",
            "wrapped_non_json_response": True,
        },
        text=text,
        variant="vlm_text_wrapped",
    )
    return _normalize_controlled_candidate(candidate, sample, config)


def _normalize_controlled_candidate(
    candidate: ScriptCandidate,
    sample: VideoSample,
    config: GenerationConfig,
) -> ScriptCandidate:
    """Keep VLM wording but enforce the task's timeline/control contract."""
    target_segments = max(1, int(config.segments or len(candidate.timeline) or 1))
    timeline = list(candidate.timeline[:target_segments])
    while len(timeline) < target_segments:
        timeline.append(
            ScriptSegment(
                start=0.0,
                end=0.0,
                narration="",
                on_screen_text="",
                selling_point="",
                control_tags=[],
            )
        )

    step = float(config.target_duration_sec or 30) / target_segments
    cta_index = _expected_cta_index(target_segments, config.cta_position)
    points = [point for point in (sample.selling_points or config.selling_points) if point]
    cta = sample.cta or "点击了解更多"

    normalized = []
    for idx, segment in enumerate(timeline):
        point = points[min(idx, len(points) - 1)] if points else segment.selling_point
        tags = [tag for tag in segment.control_tags if tag != "cta"]
        if point and "selling_point" not in tags:
            tags.append("selling_point")
        if idx == cta_index:
            tags.append("cta")
        narration = normalize_text(segment.narration)
        if not narration:
            narration = f"镜头信息聚焦{point or '核心信息'}，用真实画面支撑判断。"
        if point and point not in narration:
            narration = _append_clause(narration, f"突出{point}")
        if idx == cta_index and cta not in narration:
            narration = _append_clause(narration, f"现在{cta}")
        narration = _sanitize_script_text(narration, config.risk_terms)
        if len(narration) < 18:
            narration = _append_clause(narration, f"画面支撑{point or '关键信息'}")
        if len(narration) > 56:
            narration = narration[:55].rstrip("，,；;、 ") + "。"

        on_screen_text = normalize_text(segment.on_screen_text) or point or ("立即行动" if idx == cta_index else "核心亮点")
        on_screen_text = _sanitize_script_text(on_screen_text, config.risk_terms)

        normalized.append(
            ScriptSegment(
                start=idx * step,
                end=(idx + 1) * step,
                narration=narration,
                on_screen_text=on_screen_text,
                selling_point=point,
                control_tags=tags,
            )
        )

    candidate.timeline = normalized
    candidate.controls.update(
        {
            "target_duration_sec": config.target_duration_sec,
            "segments": target_segments,
            "cta_position": config.cta_position,
            "pace": config.pace,
            "information_density": config.information_density,
            "selling_points": points or config.selling_points,
            "risk_terms": config.risk_terms,
            "control_postprocessed": True,
        }
    )
    candidate.text = "\n".join(segment.narration for segment in normalized)
    return candidate


def _clean_model_text(content: str) -> str:
    text = normalize_text(content)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_text_segments(text: str, segment_count: int) -> list[str]:
    pieces = [
        piece.strip()
        for piece in re.split(r"(?<=[。！？!?])\s+|[\n\r]+|(?:^|\s+)\d+[.、)]\s*", text)
        if piece.strip()
    ]
    if len(pieces) >= segment_count:
        return pieces[:segment_count]
    if not text:
        return []
    window = max(1, len(text) // segment_count)
    chunks = []
    for idx in range(segment_count):
        chunk = text[idx * window : (idx + 1) * window].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _append_clause(text: str, clause: str) -> str:
    text = text.rstrip("。.!！?？；;，, ")
    return f"{text}，{clause}。"


def _sanitize_script_text(text: str, risk_terms: list[str]) -> str:
    replacements = {
        "绝对": "更",
        "第一": "突出",
        "治愈": "舒缓",
        "稳赚": "更稳妥",
        "包治": "辅助改善",
        "最低价": "优惠价",
    }
    for term in risk_terms:
        if not term:
            continue
        text = text.replace(term, replacements.get(term, ""))
    return normalize_text(text)


def _expected_cta_index(segment_count: int, position: str) -> int:
    if segment_count <= 0:
        return 0
    if position == "early":
        return min(1, segment_count - 1)
    if position == "middle":
        return segment_count // 2
    return segment_count - 1


def _candidate_id(raw_id: Any, video_id: str, idx: int) -> str:
    raw = normalize_text(raw_id) or f"vlm_{idx:02d}"
    video = normalize_text(video_id) or "video"
    if not raw.startswith(video):
        raw = f"{video}_{raw}"
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("_") or f"{video}_vlm_{idx:02d}"


def _normalize_control_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        return [tag.strip() for tag in re.split(r"[,，;；/、\s]+", value) if tag.strip()]
    if isinstance(value, list):
        return [normalize_text(tag) for tag in value if normalize_text(tag)]
    return []


def _extract_json(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", content)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _extract_json_object(content: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(content)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
