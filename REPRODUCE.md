# Reproducing the Full-Dataset Experiments

This guide describes how to reproduce the full-dataset STARS experiments for three datasets and three multimodal models. All paths and endpoints are provided through environment variables; no local machine or server-specific paths are required in the configs.

## 1. Prepare Datasets

Download the datasets from their official sources and set:

```bash
export LONGBENCH_ANNOTATION_PATH=/path/to/LongVideoBench/lvb_val.json
export LONGBENCH_VIDEO_ROOT=/path/to/LongVideoBench/videos

export CGBENCH_ANNOTATION_PATH=/path/to/CG-Bench/cgbench.json
export CGBENCH_VIDEO_ROOT=/path/to/CG-Bench

export VIDEOMME_ANNOTATION_PATH=/path/to/Video-MME/video_mme_annotations_prq.json
export VIDEOMME_VIDEO_ROOT=/path/to/Video-MME
```

The full-dataset configs use:

- LongVideoBench: all rows in `LONGBENCH_ANNOTATION_PATH`.
- CG-Bench: all rows in `CGBENCH_ANNOTATION_PATH`.
- Video-MME: all rows in `VIDEOMME_ANNOTATION_PATH`.

No sampled-subset setting is used in the open-source full configs.

## 2. Prepare Model Checkpoints

Download checkpoints from the official model repositories. The serving wrappers expose each model as an OpenAI-compatible `/v1/chat/completions` endpoint.

Example commands:

```bash
python scripts/serve_llava_video_openai.py \
  --model-path /path/to/LLaVA-Video-7B-Qwen2 \
  --model-name llava_qwen \
  --served-model-name LLaVA-Video-7B-Qwen2 \
  --host <host> \
  --port <port> \
  --max-frames 16
```

```bash
python scripts/serve_internvl_openai.py \
  --model-path /path/to/InternVL2_5-8B \
  --served-model-name InternVL2.5-8B \
  --host <host> \
  --port <port> \
  --max-frames 16
```

```bash
python scripts/serve_videollama2_openai.py \
  --model-path /path/to/VideoLLaMA2-7B-16F \
  --repo-path /path/to/VideoLLaMA2 \
  --served-model-name VideoLLaMA2-7B-16F \
  --host <host> \
  --port <port> \
  --max-frames 16
```

Then set endpoint URLs:

```bash
export LLAVA_VIDEO_ENDPOINT_URL=http://<llava-host>:<port>/v1/chat/completions
export INTERNVL25_ENDPOINT_URL=http://<internvl-host>:<port>/v1/chat/completions
export VIDEOLLAMA2_ENDPOINT_URL=http://<videollama2-host>:<port>/v1/chat/completions
export VLM_TEMPERATURE=0.3
```

If your endpoints require authentication, set:

```bash
export VLM_API_KEY=<your-key>
```

Do not commit `.env` files or secret values.

## 3. Run the Full Matrix

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

The runner calls `scripts/run_experiment.py` with `--strict-model`, `--strict-video`, and `--resume` by default.

Important flags:

- `--limit 0`: evaluate the full annotation file.
- `--num-candidates 4`: generate four structured script candidates per sample.
- `--max-frames 16`: use sparse visual observations with up to sixteen frames.
- `--text-reward none`: disable text-only reward models.
- `--no-ranker`: skip pairwise ranker training.
- `--resume`: continue from existing `results.jsonl` if interrupted.

## 4. Run One Config Manually

Each file under `configs/full/` can be run directly. For example:

```bash
python scripts/run_experiment.py \
  --config configs/full/longvideobench__llava_video_qwen2.json \
  --strict-model \
  --strict-video \
  --resume \
  --no-ranker
```

## 5. Check Results

After a run, inspect:

```bash
cat outputs/full/summary.md
find outputs/full -name metrics_report.md -print
```

Each model-dataset directory contains:

- `results.jsonl`
- `metrics.json`
- `metrics_report.md`
- `preference_pairs.jsonl`

## 6. Reproducibility Notes

- Set the same model checkpoint versions and decoding settings when comparing numbers.
- The default generation temperature is `0.3`.
- The code uses a fixed seed of `42`, but model-server implementations may still have small nondeterminism depending on backend kernels and sampling.
- The released full configs do not include local paths, server addresses, generated logs, dataset files, or model weights.
