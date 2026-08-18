from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.checkpoint_identity import (
    build_local_checkpoint_manifest,
    write_local_checkpoint_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hash a local model checkpoint and write a STARS identity manifest."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--revision",
        required=True,
        help="Immutable Hugging Face commit or documented acquisition identifier.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_local_checkpoint_manifest(
        args.model_path,
        model_id=args.model_id,
        revision=args.revision,
    )
    output = write_local_checkpoint_manifest(args.output, manifest)
    print(f"checkpoint_identity_sha256: {manifest['identity_sha256']}")
    print(f"hashed_files: {manifest['file_count']}")
    print(f"hashed_bytes: {manifest['total_bytes']}")
    print(f"manifest: {output}")


if __name__ == "__main__":
    main()
