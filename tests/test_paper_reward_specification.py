from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.config import GenerationConfig
from creative_video_exp.generation import RUNNER_VARIANT, ScriptCandidate, ScriptSegment
from creative_video_exp.representations import _aggregation_weights
from creative_video_exp.reward import (
    RewardBreakdown,
    _readability_score,
    _rhythm_score,
    validate_candidate,
)


def _candidate(
    lengths: list[int],
    durations: list[float],
    salient_points: list[str] | None = None,
) -> ScriptCandidate:
    timeline: list[ScriptSegment] = []
    start = 0.0
    points = salient_points or [""] * len(lengths)
    for index, (length, duration) in enumerate(zip(lengths, durations, strict=True)):
        narration = "a" * max(0, length - 1) + "."
        end = start + duration
        timeline.append(
            ScriptSegment(
                start=start,
                end=end,
                narration=narration,
                on_screen_text="",
                salient_point=points[index],
                control_tags=["summary"] if index == len(lengths) - 1 else [],
            )
        )
        start = end
    return ScriptCandidate(
        candidate_id="candidate",
        timeline=timeline,
        controls={},
        text="",
        variant=RUNNER_VARIANT,
        missing_required_fields=[],
    )


class PaperRewardSpecificationTest(unittest.TestCase):
    def test_reward_serialization_preserves_selection_precision(self) -> None:
        reward = RewardBreakdown(
            total=0.5000004,
            alignment=0.5000003,
            readability=0.5000002,
            rhythm=0.5000001,
            control=1.0,
            risk=1.0,
            violations=[],
            risk_terms_detected=[],
        )
        payload = reward.as_dict()
        self.assertEqual(payload["total"], 0.5000004)
        self.assertEqual(payload["alignment"], 0.5000003)

    def test_readability_uses_character_length_balance_and_completion_bonus(self) -> None:
        config = GenerationConfig()
        candidate = _candidate([60, 66, 72, 78, 84], [6, 6, 6, 6, 6])
        target = 72.0
        expected = (
            1.0
            - np.mean(np.abs(np.asarray([60, 66, 72, 78, 84]) - target) / target)
            - 0.35 * np.std([60, 66, 72, 78, 84]) / target
            + 0.08
        )
        actual = _readability_score(candidate, config, beta=0.35, sentence_bonus=0.08)
        self.assertAlmostEqual(actual, float(expected), places=12)

    def test_rhythm_combines_evenness_and_uniform_pace_profile(self) -> None:
        config = GenerationConfig()
        durations = [4, 5, 6, 7, 8]
        candidate = _candidate([72] * 5, durations)
        evenness = 1.0 - np.std(durations) / (6.0 + 1e-8)
        pace = 1.0 - np.mean(np.abs(np.asarray(durations) - 6.0) / (6.0 + 1e-8))
        expected = 0.55 * evenness + 0.45 * pace
        actual = _rhythm_score(
            candidate,
            config,
            evenness_balance=0.55,
            epsilon=1e-8,
        )
        self.assertAlmostEqual(actual, float(expected), places=12)

    def test_explicit_pace_profile_is_normalized_to_target_duration(self) -> None:
        config = GenerationConfig(
            pace="dynamic",
            pace_duration_profile=[1, 2, 3, 4, 5],
        )
        candidate = _candidate([72] * 5, [2, 4, 6, 8, 10])
        evenness = 1.0 - np.std([2, 4, 6, 8, 10]) / (6.0 + 1e-8)
        expected = 0.55 * evenness + 0.45
        actual = _rhythm_score(
            candidate,
            config,
            evenness_balance=0.55,
            epsilon=1e-8,
        )
        self.assertAlmostEqual(actual, float(expected), places=12)

    def test_control_requires_gapless_timestamps_and_key_information_order(self) -> None:
        config = GenerationConfig(salient_points=["first event", "second event"])
        candidate = _candidate(
            [72] * 5,
            [6] * 5,
            ["first event", "second event", "", "", ""],
        )
        self.assertNotIn("timestamp_validity", validate_candidate(candidate, config))
        self.assertNotIn("key_information_order", validate_candidate(candidate, config))

        candidate.timeline[0].salient_point = "second event"
        candidate.timeline[1].salient_point = "first event"
        self.assertIn("key_information_order", validate_candidate(candidate, config))

        candidate.timeline[1].start = 6.25
        self.assertIn("timestamp_validity", validate_candidate(candidate, config))

    def test_empty_key_information_control_passes(self) -> None:
        config = GenerationConfig(salient_points=[])
        candidate = _candidate([72] * 5, [6] * 5)
        self.assertNotIn("key_information_order", validate_candidate(candidate, config))

    def test_first_frame_novelty_is_one(self) -> None:
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        weights = _aggregation_weights(
            embeddings,
            centrality_weight=0.55,
            temperature=0.35,
        )
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
