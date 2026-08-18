from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.config import DataConfig
from creative_video_exp.data import load_samples
from creative_video_exp.semantic_points import FORMAL_REFERENCE_PROTOCOL
from creative_video_exp.utils import normalize_text


REFERENCE_PROTOCOL = FORMAL_REFERENCE_PROTOCOL
QUESTION_FIELDS = ("question", "query", "problem")
DIRECT_ANSWER_FIELDS = (
    "answer",
    "gt_answer",
    "label",
    "golden_answer",
    "correct_answer",
)
ANSWER_SELECTOR_FIELDS = (
    "correct_choice",
    "right_answer",
    "answer_idx",
    "answer_index",
    "correct_idx",
    "correct_option",
)
CHOICE_FIELDS = ("choices", "candidates", "options")
ROW_KEYS = ("data", "samples", "annotations", "items", "questions", "qa", "qas")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a formal manifest with evaluation-only Semantic Point Coverage "
            "references derived deterministically from official QA annotations."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset label for provenance.")
    parser.add_argument(
        "--input",
        required=True,
        help="Complete row-level annotation JSON path derived from the official data.",
    )
    parser.add_argument("--output", required=True, help="New SPC manifest JSON path.")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Permit replacing an existing generated output manifest.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_field(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, Any]:
    for field in fields:
        value = row.get(field)
        if value not in (None, "", [], {}):
            return field, value
    return "", ""


def _rows(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(payload, list):
        rows = payload
        row_key = None
    elif isinstance(payload, dict):
        row_key = next(
            (key for key in ROW_KEYS if isinstance(payload.get(key), list)),
            None,
        )
        if row_key is None:
            raise ValueError(
                "SPC input must be a JSON list or an object containing a common "
                "top-level annotation list."
            )
        rows = payload[row_key]
    else:
        raise ValueError(
            "SPC input must be a JSON list or an object containing annotations."
        )
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every annotation row must be a JSON object.")
    return [dict(row) for row in rows], row_key


def build_reference_points(question: str, answer: str) -> list[str]:
    question = normalize_text(question)
    answer = normalize_text(answer)
    if not question:
        raise ValueError("Official question text is empty.")
    if not answer:
        raise ValueError("Official correct-answer text is empty.")
    return [
        f"Evidence context: {question}",
        f"Expected answer: {answer}",
    ]


def build_manifest(input_path: Path, dataset: str) -> tuple[Any, dict[str, Any]]:
    input_path = input_path.expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows, row_key = _rows(payload)
    samples = load_samples(
        DataConfig(name=dataset, annotation_path=str(input_path)),
        project_root=PROJECT_ROOT,
    )
    if len(rows) != len(samples):
        raise RuntimeError(
            f"Manifest row/sample mismatch: {len(rows)} rows versus {len(samples)} samples."
        )

    output_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, (row, sample) in enumerate(zip(rows, samples), start=1):
        if row.get("semantic_points"):
            failures.append(
                f"row {index} ({sample.video_id}): canonical semantic_points already exist"
            )
            continue

        question_field, question_value = _first_field(row, QUESTION_FIELDS)
        direct_answer_field, _ = _first_field(row, DIRECT_ANSWER_FIELDS)
        selector_field, _ = _first_field(row, ANSWER_SELECTOR_FIELDS)
        choice_field, choices = _first_field(row, CHOICE_FIELDS)
        answer_source_field = direct_answer_field or selector_field

        try:
            points = build_reference_points(
                normalize_text(question_value),
                sample.answer,
            )
        except ValueError as exc:
            failures.append(f"row {index} ({sample.video_id}): {exc}")
            continue

        if choice_field and isinstance(choices, (list, dict)):
            raw_selector = row.get(answer_source_field)
            if (
                isinstance(raw_selector, str)
                and len(raw_selector.strip()) == 1
                and raw_selector.strip().isalpha()
                and sample.answer.strip().upper() == raw_selector.strip().upper()
            ):
                failures.append(
                    f"row {index} ({sample.video_id}): answer selector was not expanded"
                )
                continue

        annotated = dict(row)
        annotated["semantic_points"] = points
        annotated["semantic_point_provenance"] = {
            "protocol": REFERENCE_PROTOCOL,
            "reference_usage": "evaluation_only",
            "reference_enters_generation": False,
            "reference_enters_reward": False,
            "reference_enters_candidate_selection": False,
            "question_source_field": question_field,
            "answer_source_field": answer_source_field,
            "choice_source_field": choice_field,
            "resolved_answer_text": sample.answer,
            "reference_templates": [
                "Evidence context: {official_question}",
                "Expected answer: {resolved_official_correct_answer}",
            ],
        }
        output_rows.append(annotated)

    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(
            f"Cannot build formal SPC references for {len(failures)}/{len(rows)} rows:\n"
            f"{preview}"
        )

    protocol = {
        "name": REFERENCE_PROTOCOL,
        "dataset": dataset,
        "source_manifest": input_path.name,
        "source_manifest_sha256": _sha256(input_path),
        "construction": "deterministic_template_from_official_qa_annotations",
        "semantic_points_per_sample": 2,
        "reference_templates": [
            "Evidence context: {official_question}",
            "Expected answer: {resolved_official_correct_answer}",
        ],
        "answer_resolution": (
            "Direct official answer text, or the official answer selector expanded "
            "against choices/candidates/options."
        ),
        "reference_usage": "evaluation_only",
        "reference_enters_generation": False,
        "reference_enters_reward": False,
        "reference_enters_candidate_selection": False,
        "builder_version": "stars",
        "builder_script_sha256": _sha256(Path(__file__).resolve()),
    }
    if row_key is None:
        output_payload = {
            "samples": output_rows,
            "semantic_point_reference_protocol": protocol,
        }
    else:
        output_payload = dict(payload)
        output_payload[row_key] = output_rows
        output_payload["semantic_point_reference_protocol"] = protocol
    summary = {
        "dataset": dataset,
        "source_manifest": input_path.name,
        "source_manifest_sha256": _sha256(input_path),
        "reference_protocol": REFERENCE_PROTOCOL,
        "samples": len(output_rows),
        "semantic_points_per_sample": 2,
        "total_semantic_points": 2 * len(output_rows),
    }
    return output_payload, summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("--output must differ from --input; source manifests are immutable.")
    if output_path.exists() and not args.overwrite_output:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite-output only after "
            "confirming the target is a generated SPC manifest."
        )

    output_payload, summary = build_manifest(input_path, args.dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["output"] = str(output_path)
    summary["output_sha256"] = _sha256(output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
