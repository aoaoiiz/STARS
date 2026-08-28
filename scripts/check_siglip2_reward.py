from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.representations import Siglip2FrameEncoder, script_video_alignment
from creative_video_exp.video import SparseFrameBatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and validate the frozen SigLIP2 formal reward encoder."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("REWARD_VISION_MODEL_PATH", ""),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path:
        raise RuntimeError(
            "Provide --model-path or set REWARD_VISION_MODEL_PATH."
        )
    encoder = Siglip2FrameEncoder(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
    )
    frames = _diagnostic_frames()
    batch = SparseFrameBatch(
        frames=frames,
        selected_indices=list(range(len(frames))),
        bin_ids=list(range(len(frames))),
        source_kind="diagnostic",
    )
    representation = encoder.encode(
        batch,
        context_text="",
    )
    score = script_video_alignment(
        "A presenter uses visible details to explain the product.",
        representation,
        visual_grounding_balance=0.5,
        text_anchor_semantic_balance=0.7,
    )
    report = dict(encoder.model_report)
    report.update(
        {
            "encoded_frames": len(representation.frame_embeddings),
            "selected_tags": representation.content_tags,
            "diagnostic_alignment_score": round(score, 6),
            "status": "ok",
        }
    )
    print(json.dumps(report, indent=2))


def _diagnostic_frames() -> np.ndarray:
    height = width = 224
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    frames = []
    for index in range(4):
        frame = np.stack(
            [
                np.roll(xx, index * 12, axis=1),
                np.roll(yy, index * 12, axis=0),
                ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8),
            ],
            axis=-1,
        )
        frames.append(frame)
    return np.stack(frames, axis=0)


if __name__ == "__main__":
    main()
