from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DataConfig
from .utils import as_list, iter_jsonl, normalize_text, resolve_path


@dataclass
class VideoSample:
    video_id: str
    video_path: str = ""
    caption: str = ""
    question: str = ""
    answer: str = ""
    category: str = ""
    duration: float | None = None
    semantic_points: list[str] = field(default_factory=list)
    semantic_point_source_field: str = ""
    target_audience: str = ""
    closing_summary: str = ""
    evaluation_references: list[dict[str, Any]] = field(default_factory=list)
    source_row_fingerprints: list[str] = field(default_factory=list)
    source_annotation_row_count: int = 1
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def context_text(self) -> str:
        parts = [
            self.caption,
            self.question,
            self.answer,
            self.category,
            self.target_audience,
        ]
        return " ".join(part for part in parts if part)

    @property
    def reference_text(self) -> str:
        parts = [
            self.caption,
            self.question,
            self.answer,
            " ".join(self.semantic_points),
            self.category,
        ]
        return " ".join(part for part in parts if part)


def load_samples(config: DataConfig, project_root: str | Path | None = None) -> list[VideoSample]:
    rows = _load_rows(config, project_root)
    if config.sample_unit == "annotation_row":
        rows = _slice_rows(rows, config.offset, config.limit)
        samples = [_row_to_sample(row, config, project_root) for row in rows]
    elif config.sample_unit == "unique_video":
        row_samples = [
            (_row_fingerprint(row), _row_to_sample(row, config, project_root))
            for row in rows
        ]
        row_samples = [item for item in row_samples if item[1].video_id]
        samples = _merge_unique_video_samples(row_samples)
        samples = _slice_rows(samples, config.offset, config.limit)
    else:
        raise ValueError(
            "data.sample_unit must be `annotation_row` or `unique_video`."
        )
    samples = [sample for sample in samples if sample.video_id]
    if not samples:
        raise RuntimeError("No samples were loaded. Check annotation_path, hf_dataset, or limit.")
    return samples


def _load_rows(config: DataConfig, project_root: str | Path | None) -> list[dict[str, Any]]:
    if config.annotation_path:
        path = resolve_path(config.annotation_path, project_root)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")
        return _load_local_annotation(path)

    if config.hf_dataset:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Install `datasets` or provide a local annotation_path."
            ) from exc
        dataset = load_dataset(config.hf_dataset, config.hf_config, split=config.split)
        return [dict(row) for row in dataset]

    raise ValueError("Either annotation_path or hf_dataset must be provided.")


def _load_local_annotation(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return list(iter_jsonl(path))
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        if isinstance(payload, dict):
            for key in ("data", "samples", "annotations", "items", "questions", "qa", "qas"):
                if isinstance(payload.get(key), list):
                    return [dict(item) for item in payload[key]]
            return [payload]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported annotation file suffix: {suffix}")


def _row_to_sample(row: dict[str, Any], config: DataConfig, project_root: str | Path | None) -> VideoSample:
    video_id = _first(
        row,
        "video_id",
        "videoID",
        "video_uid",
        "video_name",
        "video",
        "id",
        "uid",
        "youtube_id",
        "name",
    )
    video_path = _first(
        row,
        "video_path",
        "video_file",
        "video_filename",
        "path",
        "filepath",
        "file",
        "url",
        "video_url",
    )
    video_roots = _resolve_video_roots(config, project_root)
    if video_path:
        video_path = _resolve_video_path(str(video_path), video_roots, project_root)
    elif video_roots and video_id:
        video_path = _guess_video_path(video_roots, str(video_id), config.video_search_dirs)

    caption = " ".join(
        part
        for part in [
            _first(row, "caption", "description", "desc", "summary"),
            _format_subtitles(_first(row, "subtitle", "subtitles", "transcript", "asr", "audio_transcript")),
            _first(row, "prompt", "instruction", "query", "clue", "evidence"),
        ]
        if part
    )
    question = _format_question(row)
    choices = _choice_values(row)
    direct_answer = normalize_text(
        _first(row, "answer", "gt_answer", "label", "golden_answer", "correct_answer")
    )
    selector_answer = _derive_answer_from_choices(row)
    answer = selector_answer or _expand_choice_answer(direct_answer, choices)
    category = normalize_text(
        _first(
            row,
            "category",
            "question_category",
            "topic_category",
            "domain",
            "sub_category",
            "subfield",
            "task_type",
            "type",
            "video_duration_type",
        )
    )
    duration = _safe_float(
        _first(row, "duration", "duration_sec", "video_duration", "length", "duration_seconds")
    )
    semantic_point_source_field, semantic_point_payload = _first_with_key(
        row,
        "semantic_points",
        "semantic_point",
        "expected_semantic_points",
        "evidence_points",
        "key_points",
        "key_information",
        "highlights",
        "clues",
        "clue",


        "salient_points",
        "salient_point",
        "selling_points",
        "selling_point",
        "product_points",
    )
    semantic_points = as_list(semantic_point_payload)
    target_audience = normalize_text(_first(row, "target_audience", "audience", "user_group"))
    closing_summary = normalize_text(_first(row, "closing_summary", "closing", "conclusion"))

    return VideoSample(
        video_id=normalize_text(video_id),
        video_path=video_path,
        caption=normalize_text(caption),
        question=question,
        answer=answer,
        category=category,
        duration=duration,
        semantic_points=semantic_points,
        semantic_point_source_field=semantic_point_source_field,
        target_audience=target_audience,
        closing_summary=closing_summary,
        evaluation_references=[
            {
                "question": question,
                "answer": answer,
                "caption": normalize_text(caption),
                "category": category,
                "semantic_points": list(semantic_points),
                "semantic_point_source_field": semantic_point_source_field,
            }
        ],
        source_row_fingerprints=[_row_fingerprint(row)],
        source_annotation_row_count=1,
        raw=row,
    )


def _merge_unique_video_samples(
    row_samples: list[tuple[str, VideoSample]],
) -> list[VideoSample]:
    id_to_paths: dict[str, set[str]] = {}
    path_to_ids: dict[str, set[str]] = {}
    for _, sample in row_samples:
        video_id = normalize_text(sample.video_id)
        video_path = _canonical_video_path(sample.video_path)
        if video_id and video_path:
            id_to_paths.setdefault(video_id, set()).add(video_path)
            path_to_ids.setdefault(video_path, set()).add(video_id)

    conflicting_ids = {
        video_id: sorted(paths)
        for video_id, paths in id_to_paths.items()
        if len(paths) > 1
    }
    if conflicting_ids:
        raise RuntimeError(
            "A video_id resolves to multiple video paths: "
            f"{list(conflicting_ids.items())[:5]}."
        )
    conflicting_paths = {
        video_path: sorted(video_ids)
        for video_path, video_ids in path_to_ids.items()
        if len(video_ids) > 1
    }
    if conflicting_paths:
        raise RuntimeError(
            "A video path is associated with multiple video_ids: "
            f"{list(conflicting_paths.items())[:5]}."
        )

    groups: dict[tuple[str, str], list[tuple[str, VideoSample]]] = {}
    for fingerprint, sample in row_samples:
        video_id = normalize_text(sample.video_id)
        video_path = _canonical_video_path(sample.video_path)
        if not video_path and video_id in id_to_paths:
            video_path = next(iter(id_to_paths[video_id]))
        key = ("path", video_path) if video_path else ("video_id", video_id)
        groups.setdefault(key, []).append((fingerprint, sample))

    return [_merge_video_group(key, group) for key, group in groups.items()]


def _merge_video_group(
    key: tuple[str, str],
    group: list[tuple[str, VideoSample]],
) -> VideoSample:
    first = group[0][1]
    video_ids = _ordered_unique(
        normalize_text(sample.video_id) for _, sample in group if sample.video_id
    )
    if len(video_ids) != 1:
        raise RuntimeError(
            f"Conflicting video_ids for {key[1]}: {video_ids}."
        )

    paths = _ordered_unique(
        _canonical_video_path(sample.video_path)
        for _, sample in group
        if sample.video_path
    )
    if len(paths) > 1:
        raise RuntimeError(
            f"Conflicting video paths for {video_ids[0]}: {paths}."
        )

    durations = [
        float(sample.duration)
        for _, sample in group
        if sample.duration is not None
    ]
    if durations and any(
        not math.isclose(value, durations[0], rel_tol=1e-6, abs_tol=1e-3)
        for value in durations[1:]
    ):
        raise RuntimeError(
            f"Conflicting durations for {video_ids[0]}: {durations}."
        )

    references: list[dict[str, Any]] = []
    reference_positions: dict[str, int] = {}
    for fingerprint, sample in group:
        reference = dict(sample.evaluation_references[0])
        reference["source_row_fingerprints"] = [fingerprint]
        reference_key = _evaluation_reference_key(reference)
        if reference_key in reference_positions:
            existing = references[reference_positions[reference_key]]
            existing["source_row_fingerprints"].append(fingerprint)
            existing["caption"] = "\n".join(
                _ordered_unique(
                    [
                        normalize_text(existing.get("caption", "")),
                        normalize_text(reference.get("caption", "")),
                    ]
                )
            )
            existing["category"] = " | ".join(
                _ordered_unique(
                    [
                        normalize_text(existing.get("category", "")),
                        normalize_text(reference.get("category", "")),
                    ]
                )
            )
        else:
            reference_positions[reference_key] = len(references)
            references.append(reference)

    semantic_points = [
        point
        for reference in references
        for point in reference.get("semantic_points", [])
    ]
    semantic_sources = _ordered_unique(
        str(reference.get("semantic_point_source_field", ""))
        for reference in references
        if reference.get("semantic_points")
    )
    semantic_source = semantic_sources[0] if len(semantic_sources) == 1 else ""
    captions = _ordered_unique(
        normalize_text(sample.caption)
        for _, sample in group
        if sample.caption
    )
    categories = _ordered_unique(
        normalize_text(sample.category)
        for _, sample in group
        if sample.category
    )
    questions = [
        normalize_text(reference.get("question", ""))
        for reference in references
        if reference.get("question")
    ]
    answers = [
        normalize_text(reference.get("answer", ""))
        for reference in references
        if reference.get("answer")
    ]
    fingerprints = [fingerprint for fingerprint, _ in group]

    raw = dict(first.raw)
    raw.update(
        {
            "video_id": video_ids[0],
            "video_path": paths[0] if paths else "",
            "caption": "\n".join(captions),
            "question": "\n".join(questions),
            "answer": "\n".join(answers),
            "category": " | ".join(categories),
            "semantic_points": list(semantic_points),
            "evaluation_references": references,
            "source_row_fingerprints": fingerprints,
            "source_annotation_row_count": len(group),
        }
    )
    provenance = _merge_semantic_point_provenance(group, references, fingerprints)
    if provenance:
        raw["semantic_point_provenance"] = provenance

    return VideoSample(
        video_id=video_ids[0],
        video_path=paths[0] if paths else "",
        caption="\n".join(captions),
        question="\n".join(questions),
        answer="\n".join(answers),
        category=" | ".join(categories),
        duration=durations[0] if durations else None,
        semantic_points=semantic_points,
        semantic_point_source_field=semantic_source,
        target_audience="\n".join(
            _ordered_unique(
                normalize_text(sample.target_audience)
                for _, sample in group
                if sample.target_audience
            )
        ),
        closing_summary="\n".join(
            _ordered_unique(
                normalize_text(sample.closing_summary)
                for _, sample in group
                if sample.closing_summary
            )
        ),
        evaluation_references=references,
        source_row_fingerprints=fingerprints,
        source_annotation_row_count=len(group),
        raw=raw,
    )


def _merge_semantic_point_provenance(
    group: list[tuple[str, VideoSample]],
    references: list[dict[str, Any]],
    fingerprints: list[str],
) -> dict[str, Any]:
    provenances = [
        sample.raw.get("semantic_point_provenance")
        for _, sample in group
        if isinstance(sample.raw.get("semantic_point_provenance"), dict)
    ]
    if not provenances:
        return {}
    merged = dict(provenances[0])
    for field in (
        "protocol",
        "reference_usage",
        "reference_enters_generation",
        "reference_enters_reward",
        "reference_enters_candidate_selection",
    ):
        values = {json.dumps(item.get(field), sort_keys=True) for item in provenances}
        if len(values) > 1:
            raise RuntimeError(
                f"Conflicting semantic-point provenance field `{field}`."
            )
    merged.update(
        {
            "qa_reference_count": len(references),
            "source_annotation_row_count": len(group),
            "source_row_fingerprints": fingerprints,
        }
    )
    return merged


def _evaluation_reference_key(reference: dict[str, Any]) -> str:
    payload = {
        "question": normalize_text(reference.get("question", "")),
        "answer": normalize_text(reference.get("answer", "")),
        "semantic_points": [
            normalize_text(point) for point in reference.get("semantic_points", [])
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ordered_unique(values) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _canonical_video_path(video_path: str) -> str:
    if not video_path:
        return ""
    return str(Path(video_path).expanduser().resolve())


def _row_fingerprint(row: dict[str, Any]) -> str:
    payload = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _slice_rows(rows: list[dict[str, Any]], offset: int, limit: int) -> list[dict[str, Any]]:
    start = max(0, int(offset or 0))
    if limit and limit > 0:
        return rows[start : start + int(limit)]
    return rows[start:]


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return ""


def _first_with_key(row: dict[str, Any], *keys: str) -> tuple[str, Any]:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return key, row[key]
    return "", ""


def _format_question(row: dict[str, Any]) -> str:
    question = normalize_text(_first(row, "question", "query", "problem"))
    options = _choice_values(row)
    option_text = _format_options(options)
    if option_text and question:
        return f"{question} Options: {option_text}"
    return question


def _derive_answer_from_choices(row: dict[str, Any]) -> str:
    choices = _choice_values(row)
    if not isinstance(choices, (dict, list)) or not choices:
        return ""
    correct = _first(
        row,
        "correct_choice",
        "right_answer",
        "answer_idx",
        "answer_index",
        "correct_idx",
        "correct_option",
    )
    if correct in (None, ""):
        return ""
    try:
        return _expand_choice_answer(normalize_text(correct), choices)
    except (TypeError, ValueError):
        return ""
    return ""


def _choice_values(row: dict[str, Any]) -> Any:
    choices = _first(row, "choices", "candidates", "options")
    if isinstance(choices, (dict, list)) and choices:
        return choices
    indexed = []
    for idx in range(26):
        key = f"option{idx}"
        if key not in row:
            break
        value = row.get(key)
        if value not in (None, ""):
            indexed.append(value)
    return indexed


def _expand_choice_answer(answer: str, choices: Any) -> str:
    if not answer or not isinstance(choices, (dict, list)):
        return answer
    selector_label = (
        answer.strip().upper()
        if isinstance(answer, str)
        and len(answer.strip()) == 1
        and answer.strip().isalpha()
        else ""
    )
    if isinstance(choices, dict):
        if answer in choices:
            return _strip_matching_choice_label(
                normalize_text(choices[answer]), selector_label
            )
        upper = answer.upper()
        if upper in choices:
            return _strip_matching_choice_label(
                normalize_text(choices[upper]), upper
            )
        return answer
    if selector_label:
        index = ord(selector_label) - ord("A")
    else:
        try:
            index = int(answer)
        except (TypeError, ValueError):
            return answer
    if 0 <= index < len(choices):
        return _strip_matching_choice_label(
            normalize_text(choices[index]), selector_label
        )
    return answer


def _strip_matching_choice_label(text: str, selector_label: str) -> str:


    text = normalize_text(text)
    if not text or not selector_label:
        return text
    label = re.escape(selector_label.upper())
    patterns = (
        rf"^\s*{label}\s*[.):\-]\s+",
        rf"^\s*[([]\s*{label}\s*[)\]]\s*[.):\-]?\s*",
    )
    for pattern in patterns:
        cleaned, count = re.subn(pattern, "", text, count=1, flags=re.IGNORECASE)
        if count and cleaned.strip():
            return normalize_text(cleaned)
    return text


def _format_options(options: Any) -> str:
    if isinstance(options, dict):
        parts = []
        for key in sorted(options):
            value = normalize_text(options[key])
            if value:
                parts.append(f"{key}. {value}")
        return " ".join(parts)
    if isinstance(options, list):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        parts = []
        for idx, value in enumerate(options):
            text = normalize_text(value)
            if text:
                label = labels[idx] if idx < len(labels) else str(idx)
                parts.append(f"{label}. {text}")
        return " ".join(parts)
    return normalize_text(options)


def _format_subtitles(subtitles: Any) -> str:
    if isinstance(subtitles, list):
        chunks = []
        for item in subtitles:
            if isinstance(item, dict):
                text = _first(item, "text", "caption", "subtitle", "sentence", "content")
                start = _first(item, "start", "start_time", "from")
                if text and start not in (None, ""):
                    chunks.append(f"[{start}] {normalize_text(text)}")
                elif text:
                    chunks.append(normalize_text(text))
            else:
                chunks.append(normalize_text(item))
        return " ".join(chunk for chunk in chunks if chunk)
    return normalize_text(subtitles)


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_video_roots(config: DataConfig, project_root: str | Path | None) -> list[Path]:
    roots = []
    if config.video_root:
        roots.append(config.video_root)
    roots.extend(config.video_roots)
    resolved = []
    seen = set()
    for root in roots:
        path = resolve_path(root, project_root)
        if path not in seen:
            resolved.append(path)
            seen.add(path)
    return resolved


def _resolve_video_path(
    video_path: str,
    video_roots: list[Path],
    project_root: str | Path | None,
) -> str:
    candidate = Path(video_path)
    if candidate.is_absolute():
        return str(candidate)
    for root in video_roots:
        rooted = root / candidate
        if rooted.exists():
            return str(rooted)
    if video_roots:
        return str(video_roots[0] / candidate)
    return str(resolve_path(video_path, project_root))


def _guess_video_path(roots: list[Path], video_id: str, search_dirs: list[str]) -> str:
    stems = [video_id]
    if "." in video_id:
        stems.append(Path(video_id).stem)
    suffixes = ("", ".mp4", ".webm", ".mov", ".mkv", ".avi", ".npz")
    for root in roots:
        for search_dir in search_dirs or [""]:
            base = root / search_dir if search_dir else root
            for stem in stems:
                for suffix in suffixes:
                    candidate = base / f"{stem}{suffix}"
                    if candidate.exists():
                        return str(candidate)
                candidate_dir = base / stem
                if candidate_dir.exists():
                    return str(candidate_dir)
    return ""
