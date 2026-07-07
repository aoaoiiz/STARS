# Full-Dataset Configs

This directory contains 9 full-dataset experiment configs:

- 3 datasets: LongVideoBench, CG-Bench, Video-MME.
- 3 models: LLaVA-Video-7B-Qwen2, InternVL2.5-8B, VideoLLaMA2-7B-16F.

All configs use:

- `limit: 0`
- `num_candidates: 4`
- `max_frames: 16`
- `text_reasoning_weight: 0.0`
- no text-only reward model

Dataset paths and endpoint URLs are read from environment variables. See `../../REPRODUCE.md`.

