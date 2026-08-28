# Reproducing STARS

Run the commands below from the repository root.

## 1. Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-formal.txt
PYTHONPATH=src python -m unittest discover -s tests -v
```

Install the upstream LLaVA-Video, InternVL, and VideoLLaMA2 packages in the environments used to run their services.

The model services and SigLIP2 reward encoder require CUDA-capable GPUs. Start only the services needed for the selected backbone, or assign concurrent services to separate devices with sufficient memory.

## 2. Configure datasets

```bash
export LONGBENCH_ANNOTATION_PATH=/absolute/path/to/longvideobench.json
export LONGBENCH_VIDEO_ROOT=/absolute/path/to/longvideobench/videos

export CGBENCH_ANNOTATION_PATH=/absolute/path/to/cgbench.json
export CGBENCH_VIDEO_ROOT=/absolute/path/to/cgbench/videos

export VIDEOMME_ANNOTATION_PATH=/absolute/path/to/videomme.json
export VIDEOMME_VIDEO_ROOT=/absolute/path/to/videomme/videos
```

The source annotation file supplied to the builder must be a JSON object with a top-level `samples` array. Canonical evaluation-only `semantic_points` records can be constructed and checked with:

```bash
python scripts/build_semantic_point_manifests.py \
  --dataset DATASET_NAME \
  --input /absolute/path/to/source-annotations.json \
  --output /absolute/path/to/annotations-with-semantic-points.json

python scripts/audit_semantic_point_annotations.py \
  --annotation DATASET_NAME=/absolute/path/to/annotations-with-semantic-points.json \
  --require-complete \
  --require-reference-protocol official_qa_evidence_and_answer_v2
```

Set the corresponding annotation environment variable to the generated, audited manifest before running the matrix.

## 3. Register checkpoints

Create an identity sidecar for each local checkpoint:

```bash
python scripts/build_checkpoint_manifest.py \
  --model-path /absolute/path/to/checkpoint \
  --model-id MODEL_ID \
  --revision IMMUTABLE_REVISION \
  --output /absolute/path/to/checkpoint-manifest.json
```

The model identifiers are `LLaVA-Video-7B-Qwen2`, `InternVL2.5-8B`, `VideoLLaMA2-7B-16F`, and `google/siglip2-so400m-patch14-384`.

```bash
export LLAVA_VIDEO_CHECKPOINT_MANIFEST=/absolute/path/to/llava-manifest.json
export INTERNVL25_CHECKPOINT_MANIFEST=/absolute/path/to/internvl-manifest.json
export VIDEOLLAMA2_CHECKPOINT_MANIFEST=/absolute/path/to/videollama2-manifest.json

export REWARD_VISION_MODEL_PATH=/absolute/path/to/siglip2
export REWARD_VISION_CHECKPOINT_MANIFEST=/absolute/path/to/siglip2-manifest.json
export REWARD_VISION_DEVICE=cuda:0
export REWARD_VISION_DTYPE=bfloat16
```

## 4. Start model services

Start each service in its model environment.

```bash
python scripts/serve_llava_video_openai.py \
  --model-path /absolute/path/to/llava \
  --checkpoint-manifest "$LLAVA_VIDEO_CHECKPOINT_MANIFEST"
```

```bash
python scripts/serve_internvl_openai.py \
  --model-path /absolute/path/to/internvl \
  --checkpoint-manifest "$INTERNVL25_CHECKPOINT_MANIFEST"
```

```bash
python scripts/serve_videollama2_openai.py \
  --model-path /absolute/path/to/videollama2 \
  --repo-path /absolute/path/to/VideoLLaMA2 \
  --checkpoint-manifest "$VIDEOLLAMA2_CHECKPOINT_MANIFEST"
```

```bash
export LLAVA_VIDEO_ENDPOINT_URL=http://127.0.0.1:8010/v1/chat/completions
export INTERNVL25_ENDPOINT_URL=http://127.0.0.1:8011/v1/chat/completions
export VIDEOLLAMA2_ENDPOINT_URL=http://127.0.0.1:8012/v1/chat/completions
```

If the services use authentication, set `VLM_API_KEY` to the same value supplied through each service's `--api-key` argument.

## 5. Run STARS

```bash
python scripts/run_server_matrix.py
```

To run one backbone, pass one of `llava_video_qwen2`, `internvl25_8b`, or `videollama2_7b_16f` through `--models`.

The runner writes generated configurations to `outputs/stars_configs` and experiment artifacts to `outputs/stars`.

## 6. Compile evidence

```bash
python scripts/compile_revision_evidence.py \
  --matrix stars=outputs/stars \
  --output-dir outputs/stars_evidence_NEW
```

The evidence destination must not already contain files. The compiler checks completion, protocol identity, candidate-slot accounting, failure-aware metrics, and efficiency records before producing its tables.

Generated protocol and metric artifacts retain configured paths for reproducibility. Review those artifacts before distributing them outside the experiment environment.
