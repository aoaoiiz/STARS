from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.config import DataConfig
from creative_video_exp.data import load_samples
from creative_video_exp.semantic_points import formal_reference_contract_issue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit evaluation-only Semantic Point Coverage references."
    )
    parser.add_argument(
        "--annotation",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Dataset label and JSON/JSONL/CSV annotation path; repeat as needed.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero if any loaded sample lacks semantic_points.",
    )
    parser.add_argument(
        "--require-reference-protocol",
        default="",
        help="Require the top-level semantic-point reference protocol name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing_total = 0
    protocol_failures = 0
    for item in args.annotation:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, received `{item}`.")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = (
            payload.get("semantic_point_reference_protocol", {})
            if isinstance(payload, dict)
            else {}
        )
        protocol_name = protocol.get("name", "")
        samples = load_samples(
            DataConfig(
                name=name,
                annotation_path=str(path),
                sample_unit="annotation_row",
                limit=0,
            ),
            project_root=PROJECT_ROOT,
        )
        available = [sample for sample in samples if sample.semantic_points]
        missing = [sample.video_id for sample in samples if not sample.semantic_points]
        noncanonical = [
            sample.video_id
            for sample in available
            if sample.semantic_point_source_field != "semantic_points"
        ]
        invalid_contract = [
            (sample.video_id, formal_reference_contract_issue(sample.raw))
            for sample in available
            if formal_reference_contract_issue(sample.raw)
        ]
        missing_total += len(missing) + len(noncanonical) + len(invalid_contract)
        sources = Counter(
            sample.semantic_point_source_field for sample in available
        )
        print(f"\nDataset: {name}")
        print(f"annotation: {path}")
        print(f"reference_protocol: {protocol_name or 'missing'}")
        print(f"loaded_samples: {len(samples)}")
        print(f"samples_with_semantic_points: {len(available)}")
        print(f"samples_missing_semantic_points: {len(missing)}")
        print(f"samples_with_noncanonical_point_field: {len(noncanonical)}")
        print(f"samples_with_invalid_reference_contract: {len(invalid_contract)}")
        print(
            "semantic_point_reference_rate: "
            f"{len(available) / max(1, len(samples)):.6f}"
        )
        print(f"source_field_counts: {dict(sorted(sources.items()))}")
        print(f"missing_video_id_examples: {missing[:10]}")
        print(f"noncanonical_video_id_examples: {noncanonical[:10]}")
        print(f"invalid_reference_examples: {invalid_contract[:10]}")
        if (
            args.require_reference_protocol
            and protocol_name != args.require_reference_protocol
        ):
            protocol_failures += 1
            print(
                "reference_protocol_status: failed "
                f"(expected {args.require_reference_protocol})"
            )
        else:
            print("reference_protocol_status: ok")

    if (args.require_complete and missing_total) or protocol_failures:
        raise SystemExit(
            "Semantic-point audit failed: "
            f"{missing_total} samples have missing or noncanonical references; "
            f"{protocol_failures} datasets have an invalid reference protocol."
        )


if __name__ == "__main__":
    main()
