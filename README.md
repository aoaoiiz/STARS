# STARS: Structured Timeline Alignment and Rewarded Selection

STARS is a training-free framework for structured video script generation. It samples a sparse temporal observation, obtains four structured candidates from a video-language model, scores each candidate with a fixed five-component self-reward, and returns the highest-reward candidate.

This repository contains only the main experiment pipeline for LLaVA-Video-7B-Qwen2, InternVL2.5-8B, and VideoLLaMA2-7B-16F on LongVideoBench, CG-Bench, and Video-MME. The released setting is fixed to four candidates, sixteen binwise-sampled frames, English output, five timeline segments, and the frozen SigLIP2 reward encoder. Sampling, candidate-count, reward-component, and auxiliary analysis utilities are not included.

## Full-data policy

STARS processes every row in each supplied annotation file. The code contains no sample limit, offset, subset sampler, or replacement-sample mechanism. For a full-dataset experiment, each annotation path must therefore point to a complete row-level manifest derived from the official annotations, with one row per evaluation item. Nested benchmark records must be flattened without filtering before this step. Missing or undecodable videos stop the run instead of reducing the evaluation set.

Semantic Point Coverage is evaluation-only. Every row must contain a video identifier, an official question, a resolvable correct answer, and a local video reference or dataset-specific media identifier. The complete annotation manifest must contain a non-empty `semantic_points` field for every row and the required top-level reference-protocol metadata. `scripts/build_semantic_point_manifests.py` creates these fields deterministically from the official question and resolved correct answer without changing the number or order of rows.

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
patches/
  videollama2_clip_eager.patch
src/creative_video_exp/
requirements.txt
requirements-internvl.txt
```

## Installation

STARS requires Python 3.10 or later. Use one environment for the experiment runner and reward encoder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The three generation backbones should use separate environments because their upstream repositories require different dependency versions. Install [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), [InternVL](https://github.com/OpenGVLab/InternVL), and [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2) in their respective service environments. Install `requirements-internvl.txt` in the InternVL service environment. In the LLaVA-Video and VideoLLaMA2 environments, install the dependencies specified by the corresponding upstream repository together with the local service dependencies:

```bash
pip install "fastapi>=0.110" "uvicorn>=0.29" "pydantic>=2.0"
```

The released VideoLLaMA2 setting uses the eager attention backend for the CLIP vision tower. From a VideoLLaMA2 checkout at base commit `c0bb03abf6b8a6b9a8dccac006fb4db5d4d9e414`, apply the included patch before starting the service:

```bash
git apply --unidiff-zero /path/to/STARS/patches/videollama2_clip_eager.patch
```

The VideoLLaMA2 service verifies this source change at startup and refuses to run if the vision tower still forces FlashAttention2. The model checkpoint is not modified.

## Prepare full annotation manifests

Run the following command once for each complete official annotation file:

```bash
python scripts/build_semantic_point_manifests.py \
  --dataset longvideobench \
  --input /path/to/full-row-level-annotations.json \
  --output /path/to/full-annotations-with-semantic-points.json
```

Use `cgbench` and `videomme` for the other datasets. The output contains all source rows.

For Video-MME, store each downloaded video as `<videoID>.mp4` under `VIDEOMME_VIDEO_ROOT`. The loader uses the official `videoID` field for file resolution while retaining the annotation identifier as the sample identifier. HTTP and HTTPS URLs are treated as source metadata, not local paths.

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

Repeated matrix commands using the same output root merge completed model--dataset rows into the matrix summary. A newly completed row replaces only the prior row with the same dataset and model identifiers.

Use a new output directory when replacing an earlier release. Protocol identifiers and checkpoint manifests are validated before a run can resume.

## Outputs

Each model-dataset run writes `results.jsonl`, `generation_failures.jsonl`, `run_status.json`, `protocol_manifest.json`, `metrics.json`, `metrics_report.md`, and `candidate_pool_analysis/`. The matrix root also contains `summary.json` and `summary.md`.

The primary reported quantities are Reward, MV-Align, Control Success Rate, Semantic Point Coverage, CIDEr-lite, and repetition rate. Direct Generation is the first candidate. STARS selects the valid candidate with the highest composite Reward among the four requested candidate slots. Metric-wise Best@4 is reported only as a non-deployable upper bound over the same candidate pool.

Semantic Point Coverage is QA-derived in this release. Each official question and resolved correct answer yields two evaluation-only reference points. SPC is the fraction of those points whose maximum cosine similarity to a generated timeline segment reaches `0.50`. CIDEr-lite is a bounded proxy rather than the standard CIDEr implementation: it averages clipped unigram through four-gram F1 overlap against a reference formed by concatenating the annotation caption, official question, resolved answer, the two QA-derived reference points, and category. Neither metric enters generation, Reward, or candidate selection.

Failure-aware effective means retain every requested sample, assigning the documented worst-case value when a method has no valid candidate. Conditional means are also written and are labeled with the corresponding method success rate. Human-readable reports show both estimands separately. Token counts are tokenizer-specific text-token estimates and exclude visual tokens. Accounted online latency for Direct Generation includes frame sampling and the C1 request time. For STARS, it includes frame sampling, all four candidate request times, reward encoding and scoring, and selection. It excludes model loading, response parsing, and evaluation-only metric computation.

## Data and model licenses

Dataset videos, annotations, model repositories, and checkpoints are not redistributed. Obtain them from their official sources and follow their respective licenses and access conditions.
