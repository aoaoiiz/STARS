# STARS

STARS is a training-free framework for structured creative video script generation. It samples sparse temporal observations, requests four independent structured candidates from a video-language model, scores the valid candidates with a frozen multimodal reward, and selects the highest-reward candidate.

The implementation supports LongVideoBench, CG-Bench, and Video-MME with LLaVA-Video-7B-Qwen2, InternVL2.5-8B, and VideoLLaMA2-7B-16F. SigLIP2 `google/siglip2-so400m-patch14-384` supplies the frozen image and text encoders used by the reward.

## Method

Each video is represented by up to 16 frames selected from eight temporal bins. A generator requests four candidates through independent calls. Every accepted candidate contains the top-level fields `candidate_id`, `variant`, and `timeline`; each timeline item contains `start`, `end`, `narration`, `on_screen_text`, `salient_point`, and `control_tags`. STARS combines alignment, readability, rhythm, control, and risk terms into a fixed reward and returns the valid candidate with the highest score.

Semantic Point Coverage is evaluation-only. Questions, answers, captions, subtitles, transcripts, reference descriptions, and semantic-point references are not supplied to the generator or reward model.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-formal.txt
```

Generator-specific upstream packages must be installed in the environments that host the corresponding model services.

## Running experiments

Configure dataset paths, register immutable checkpoint identities, start the local model services, and run:

```bash
python scripts/run_server_matrix.py
```

Detailed commands are provided in [REPRODUCE.md](REPRODUCE.md). Dataset configuration examples are available in [configs/datasets.example.json](configs/datasets.example.json), and the Semantic Point Coverage contract is documented in [docs/SEMANTIC_POINT_COVERAGE.md](docs/SEMANTIC_POINT_COVERAGE.md).

## Outputs

Each dataset-model directory contains the protocol manifest, run status, per-video results, generation failures, preference pairs, aggregate metrics, and candidate-pool analysis. Matrix summaries are written as JSON and Markdown.

## Repository layout

- `src/creative_video_exp`: sampling, generation, reward, evaluation, provenance, and reporting
- `scripts`: experiment entry points, model services, checkpoint registration, and evidence compilation
- `configs`: dataset path examples
- `docs`: metric contracts
- `tests`: protocol and implementation tests

## License

This project is released under the MIT License.
