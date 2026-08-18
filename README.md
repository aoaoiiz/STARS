# STARS: Structured Timeline Alignment and Rewarded Selection

STARS is a training-free framework for structured video script generation. It samples a sparse temporal observation, obtains four structured candidates from a video-language model, scores each candidate with a fixed five-component self-reward, and returns the highest-reward candidate.

This repository contains only the main experiment pipeline for LLaVA-Video-7B-Qwen2, InternVL2.5-8B, and VideoLLaMA2-7B-16F on LongVideoBench, CG-Bench, and Video-MME. The released setting is fixed to four candidates, sixteen binwise-sampled frames, English output, five timeline segments, and the frozen SigLIP2 reward encoder. Sampling, candidate-count, reward-component, and auxiliary analysis utilities are not included.

## Full-data policy

STARS processes every row in each supplied annotation file. The code contains no sample limit, offset, subset sampler, or replacement-sample mechanism. For a full-dataset experiment, each annotation path must therefore point to the complete official annotation manifest. Missing or undecodable videos stop the run instead of reducing the evaluation set.

Semantic Point Coverage is evaluation-only. The complete annotation manifest must contain a non-empty `semantic_points` field for every row and the required top-level reference-protocol metadata. `scripts/build_semantic_point_manifests.py` creates these fields deterministically from the official question and resolved correct answer without changing the number or order of rows.

## Repository layout

```text
scripts/
  build_checkpoint_manifest.py
  build_semantic_point_manifests.py
  run_experiment.py
  run_server_matrix.py
  serve_internvl_openai.py
  serve_llava_video_openai.py
  serve_videollama2_openai.py
src/creative_video_exp/
requirements.txt
requirements-internvl.txt
```

## Installation

Use one environment for the experiment runner and reward encoder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The three generation backbones should use separate environments because their upstream repositories may require different dependency versions. Install the official LLaVA-Video and VideoLLaMA2 repositories in their respective environments. For InternVL2.5-8B, install `requirements-internvl.txt` in its service environment. When a VideoLLaMA2 checkout forces FlashAttention2 for the CLIP vision tower, configure that vision tower to use the eager attention backend; the model checkpoint itself is unchanged.

## Prepare full annotation manifests

Run the following command once for each complete official annotation file:

```bash
python scripts/build_semantic_point_manifests.py \
  --dataset longvideobench \
  --input /path/to/full-official-annotations.json \
  --output /path/to/full-annotations-with-semantic-points.json
```

Use `cgbench` and `videomme` for the other datasets. The output contains all source rows.

## Bind model checkpoints

Create one sidecar for each local generation checkpoint and for the SigLIP2 reward checkpoint:

```bash
python scripts/build_checkpoint_manifest.py \
  --model-path /path/to/checkpoint \
  --model-id exact-model-id \
  --revision immutable-revision \
  --output /path/to/checkpoint-manifest.json
```

The required model IDs are `LLaVA-Video-7B-Qwen2`, `InternVL2.5-8B`, `VideoLLaMA2-7B-16F`, and `google/siglip2-so400m-patch14-384`.

## Start a generation service

Each wrapper exposes an OpenAI-compatible local endpoint. The following examples use loopback addresses only; choose any free local ports.

```bash
python scripts/serve_llava_video_openai.py \
  --model-path /path/to/LLaVA-Video-7B-Qwen2 \
  --model-name llava_qwen \
  --served-model-name LLaVA-Video-7B-Qwen2 \
  --checkpoint-manifest /path/to/llava-checkpoint-manifest.json \
  --host 127.0.0.1 \
  --port 8010 \
  --max-frames 16
```

```bash
python scripts/serve_internvl_openai.py \
  --model-path /path/to/InternVL2_5-8B \
  --served-model-name InternVL2.5-8B \
  --checkpoint-manifest /path/to/internvl-checkpoint-manifest.json \
  --host 127.0.0.1 \
  --port 8011 \
  --max-frames 16 \
  --image-size 448
```

```bash
python scripts/serve_videollama2_openai.py \
  --model-path /path/to/VideoLLaMA2-7B-16F \
  --repo-path /path/to/VideoLLaMA2 \
  --served-model-name VideoLLaMA2-7B-16F \
  --checkpoint-manifest /path/to/videollama2-checkpoint-manifest.json \
  --host 127.0.0.1 \
  --port 8012 \
  --max-frames 16
```

## Configure datasets, endpoints, and reward encoder

```bash
export LONGBENCH_ANNOTATION_PATH=/path/to/full-longvideobench-manifest.json
export LONGBENCH_VIDEO_ROOT=/path/to/longvideobench-videos
export CGBENCH_ANNOTATION_PATH=/path/to/full-cgbench-manifest.json
export CGBENCH_VIDEO_ROOT=/path/to/cgbench-videos
export VIDEOMME_ANNOTATION_PATH=/path/to/full-videomme-manifest.json
export VIDEOMME_VIDEO_ROOT=/path/to/videomme-videos

export LLAVA_VIDEO_ENDPOINT_URL=http://127.0.0.1:8010/v1/chat/completions
export INTERNVL25_ENDPOINT_URL=http://127.0.0.1:8011/v1/chat/completions
export VIDEOLLAMA2_ENDPOINT_URL=http://127.0.0.1:8012/v1/chat/completions

export LLAVA_VIDEO_CHECKPOINT_MANIFEST=/path/to/llava-checkpoint-manifest.json
export INTERNVL25_CHECKPOINT_MANIFEST=/path/to/internvl-checkpoint-manifest.json
export VIDEOLLAMA2_CHECKPOINT_MANIFEST=/path/to/videollama2-checkpoint-manifest.json

export REWARD_VISION_MODEL_PATH=/path/to/siglip2-so400m-patch14-384
export REWARD_VISION_CHECKPOINT_MANIFEST=/path/to/siglip2-checkpoint-manifest.json
export REWARD_VISION_DEVICE=cuda:0
export REWARD_VISION_DTYPE=bfloat16
export VLM_TEMPERATURE=0.3
```

## Run the main experiment

If all three services are available, run the complete matrix:

```bash
python scripts/run_server_matrix.py \
  --datasets longvideobench cgbench videomme \
  --models llava_video_qwen2 internvl25_8b videollama2_7b_16f \
  --output-root outputs/full \
  --config-dir outputs/full_configs
```

On a single GPU, start one generation service at a time and run the same command with one model name. Every selected dataset is still evaluated on all rows in its annotation file.

Use a new output directory when replacing an earlier release. Protocol identifiers and checkpoint manifests are validated before a run can resume.

## Outputs

Each model-dataset run writes `results.jsonl`, `generation_failures.jsonl`, `run_status.json`, `protocol_manifest.json`, `metrics.json`, `metrics_report.md`, and `candidate_pool_analysis/`. The matrix root also contains `summary.json` and `summary.md`.

The primary reported quantities are Reward, MV-Align, Control Success Rate, Semantic Point Coverage, CIDEr-lite, and repetition rate. Direct Generation is the first candidate. STARS selects the valid candidate with the highest composite Reward among the four requested candidate slots. Metric-wise Best@4 is reported only as a non-deployable upper bound over the same candidate pool.

## Data and model licenses

Dataset videos, annotations, model repositories, and checkpoints are not redistributed. Obtain them from their official sources and follow their respective licenses and access conditions.
