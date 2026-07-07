# STARS: Structured Video Script Generation with Self-Reward Selection

STARS is an inference-time framework for structured video script generation. Given a video, STARS sparsely samples visual frames, asks a multimodal video-language model to generate multiple structured script candidates, and selects the final script with a reproducible composite self-reward.

The repository contains the experiment pipeline used for three video-language models and three video benchmarks:

- Models: LLaVA-Video-7B-Qwen2, InternVL2.5-8B, and VideoLLaMA2-7B-16F.
- Datasets: LongVideoBench, CG-Bench, and Video-MME.
- Main setting: full-dataset evaluation, `num_candidates=4`, `max_frames=16`, no text-only reward model, and no learned ranker.

Datasets, model checkpoints, generated videos, and experimental outputs are not included. Please download them from their official sources and configure paths through environment variables.

## Repository Layout

```text
configs/
  full/                    # 3 datasets x 3 models, full-dataset configs
  smoke.json               # lightweight local smoke test
scripts/
  run_experiment.py         # single config runner
  run_server_matrix.py      # full matrix runner
  serve_llava_video_openai.py
  serve_internvl_openai.py
  serve_videollama2_openai.py
src/creative_video_exp/     # STARS pipeline
sample_data/                # tiny smoke-test annotations
tests/                      # unit tests
REPRODUCE.md                # full-dataset reproduction guide
requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For real video decoding and model serving, install the optional dependencies required by your chosen model implementation, such as `transformers`, `opencv-python`, `decord`, `fastapi`, `uvicorn`, and the corresponding upstream model repositories.

## Quick Smoke Test

The smoke test uses synthetic or local lightweight frames and does not require large model checkpoints.

```bash
python scripts/run_experiment.py --config configs/smoke.json --no-ranker
python -m unittest discover tests
```

## Full-Dataset Experiments

Full-dataset configs are stored under `configs/full/`. They use `limit: 0`, which means all samples from the provided annotation file are evaluated.

Before running, set dataset paths:

```bash
export LONGBENCH_ANNOTATION_PATH=/path/to/LongVideoBench/lvb_val.json
export LONGBENCH_VIDEO_ROOT=/path/to/LongVideoBench/videos

export CGBENCH_ANNOTATION_PATH=/path/to/CG-Bench/cgbench.json
export CGBENCH_VIDEO_ROOT=/path/to/CG-Bench

export VIDEOMME_ANNOTATION_PATH=/path/to/Video-MME/video_mme_annotations_prq.json
export VIDEOMME_VIDEO_ROOT=/path/to/Video-MME
```

Set OpenAI-compatible multimodal endpoints:

```bash
export LLAVA_VIDEO_ENDPOINT_URL=http://<llava-host>:<port>/v1/chat/completions
export INTERNVL25_ENDPOINT_URL=http://<internvl-host>:<port>/v1/chat/completions
export VIDEOLLAMA2_ENDPOINT_URL=http://<videollama2-host>:<port>/v1/chat/completions
export VLM_TEMPERATURE=0.3
```

Run the full matrix:

```bash
python scripts/run_server_matrix.py \
  --datasets longvideobench cgbench videomme \
  --models llava_video_qwen2 internvl25_8b videollama2_7b_16f \
  --limit 0 \
  --num-candidates 4 \
  --max-frames 16 \
  --text-reward none \
  --no-ranker \
  --output-root outputs/full \
  --config-dir outputs/full_configs
```

See [REPRODUCE.md](REPRODUCE.md) for model-serving commands and individual config examples.

## Outputs

Each run writes:

- `results.jsonl`: sample-level candidates, reward details, selected best candidate, and metadata.
- `metrics.json`: aggregate metrics.
- `metrics_report.md`: a readable summary.
- `preference_pairs.jsonl`: self-reward preference pairs.

The key paper metrics are:

- `mean_best_reward`: composite self-reward.
- `mean_best_alignment`: multimodal video-script alignment.
- `control_success_rate_best`: structural control success rate.
- `selling_point_coverage`: selling-point coverage.
- `mean_cider_lite`: lightweight CIDEr-style reference score.
- `mean_repetition_rate`: repetition rate, lower is better.

## Notes

- STARS does not train or fine-tune the base video-language models.
- The benchmark question-answer annotations may be loaded as dataset metadata and weak references, but the task is structured video script generation rather than video question answering.
- The repository does not redistribute benchmark videos or model checkpoints. Users are responsible for following the original licenses.

