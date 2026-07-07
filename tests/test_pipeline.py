from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.config import ExperimentConfig
from creative_video_exp.data import VideoSample, load_samples
from creative_video_exp.generation import StructuredScriptGenerator
from creative_video_exp.modeling import _extract_json_object, _parse_script_candidates
from creative_video_exp.representations import StatsFrameEncoder
from creative_video_exp.reward import SelfRewardScorer
from creative_video_exp.video import SparseFrameSampler


class PipelineTest(unittest.TestCase):
    def test_smoke_pipeline_one_sample(self) -> None:
        config = ExperimentConfig.from_file(PROJECT_ROOT / "configs/smoke.json")
        config.data.limit = 1
        samples = load_samples(config.data, project_root=PROJECT_ROOT)
        sampler = SparseFrameSampler(config.sampling)
        batch = sampler.sample(samples[0].video_path, samples[0].video_id, samples[0].raw)
        self.assertEqual(len(batch.frames), config.sampling.max_frames)

        representation = StatsFrameEncoder().encode(batch, samples[0].context_text)
        self.assertTrue(representation.content_tags)
        candidates = StructuredScriptGenerator(config.generation).generate(samples[0], representation)
        self.assertEqual(len(candidates), config.generation.num_candidates)

        scorer = SelfRewardScorer(config.reward, config.generation)
        reward = scorer.score(candidates[0], representation)
        self.assertGreaterEqual(reward.total, 0.0)
        self.assertLessEqual(reward.total, 1.0)

    def test_offset_and_choice_answer_expansion(self) -> None:
        config = ExperimentConfig.from_file(PROJECT_ROOT / "configs/smoke.json")
        config.data.offset = 1
        config.data.limit = 1
        samples = load_samples(config.data, project_root=PROJECT_ROOT)
        self.assertEqual(len(samples), 1)

    def test_parse_multimodal_script_json(self) -> None:
        config = ExperimentConfig.from_file(PROJECT_ROOT / "configs/smoke.json")
        content = """
        [
          {
            "candidate_id": "cand0",
            "variant": "server",
            "timeline": [
              {
                "start": 0,
                "end": 6,
                "narration": "先用画面里的细节做开场。",
                "on_screen_text": "视觉证据",
                "selling_point": "视觉证据",
                "control_tags": ["hook", "selling_point"]
              },
              {
                "start": 6,
                "end": 12,
                "narration": "现在点击了解更多。",
                "on_screen_text": "了解更多",
                "selling_point": "行动引导",
                "control_tags": "cta"
              }
            ]
          }
        ]
        """
        config.generation.segments = 2
        sample = VideoSample(video_id="videoA", selling_points=["visual evidence"])
        candidates = _parse_script_candidates(content, sample, config.generation)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].candidate_id.startswith("videoA"))
        self.assertIn("cta", candidates[0].timeline[-1].control_tags)

    def test_parse_text_reward_json_object(self) -> None:
        payload = _extract_json_object('```json\n{"score": 0.7, "rationale": "ok"}\n```')
        self.assertIsNotNone(payload)
        self.assertEqual(payload["score"], 0.7)


if __name__ == "__main__":
    unittest.main()
