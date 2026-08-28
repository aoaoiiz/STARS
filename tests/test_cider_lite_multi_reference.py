from __future__ import annotations

import unittest
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.data import VideoSample
from creative_video_exp.generation import ScriptCandidate, ScriptSegment
from creative_video_exp.text_metrics import evaluate_generation_quality


def _candidate(text: str) -> ScriptCandidate:
    return ScriptCandidate(
        candidate_id="candidate",
        timeline=[
            ScriptSegment(
                start=0.0,
                end=30.0,
                narration=text,
                on_screen_text="",
                salient_point="",
                control_tags=["summary"],
            )
        ],
        controls={},
        text=text,
        variant="visual_story",
    )


class CiderLiteMultiReferenceTest(unittest.TestCase):
    def test_single_reference_preserves_legacy_reference_text(self) -> None:
        candidate = _candidate("alpha beta gamma delta")
        legacy = VideoSample(
            video_id="v1",
            caption="alpha beta gamma delta",
        )
        one_reference = VideoSample(
            video_id="v1",
            caption="alpha beta gamma delta",
            evaluation_references=[
                {
                    "caption": "one two three four",
                    "question": "",
                    "answer": "",
                    "category": "",
                    "semantic_points": [],
                }
            ],
        )

        legacy_quality = evaluate_generation_quality(candidate, legacy)
        one_reference_quality = evaluate_generation_quality(candidate, one_reference)

        self.assertEqual(one_reference_quality["cider_lite"], legacy_quality["cider_lite"])
        self.assertEqual(one_reference_quality["cider_lite"], 1.0)
        self.assertEqual(
            one_reference_quality["cider_lite_reference_aggregation"],
            "single_legacy_reference",
        )

    def test_multiple_references_are_scored_independently_then_averaged(self) -> None:
        candidate = _candidate("alpha beta gamma delta")
        sample = VideoSample(
            video_id="v1",
            caption="this aggregated field must not be used",
            evaluation_references=[
                {
                    "caption": "alpha beta gamma delta",
                    "question": "",
                    "answer": "",
                    "category": "",
                    "semantic_points": [],
                },
                {
                    "caption": "one two three four",
                    "question": "",
                    "answer": "",
                    "category": "",
                    "semantic_points": [],
                },
            ],
        )

        quality = evaluate_generation_quality(candidate, sample)

        self.assertEqual(quality["cider_lite"], 0.5)
        self.assertEqual(quality["cider_lite_reference_count"], 2)
        self.assertEqual(quality["cider_lite_reference_scores"], [1.0, 0.0])
        self.assertEqual(
            quality["cider_lite_reference_aggregation"],
            "equal_mean_over_evaluation_references",
        )

if __name__ == "__main__":
    unittest.main()
