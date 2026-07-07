from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.config import ExperimentConfig
from creative_video_exp.data import load_samples
from creative_video_exp.generation import StructuredScriptGenerator
from creative_video_exp.metrics import summarize_results
from creative_video_exp.modeling import (
    build_frame_encoder,
    build_script_generation_model,
    build_text_reward_model,
)
from creative_video_exp.ranker import train_pairwise_ranker
from creative_video_exp.reporting import print_metrics_summary, write_markdown_report
from creative_video_exp.reward import SelfRewardScorer, build_preference_pairs
from creative_video_exp.text_metrics import evaluate_generation_quality
from creative_video_exp.utils import ensure_dir, iter_jsonl, set_seed, stable_hash_int, write_json, write_jsonl
from creative_video_exp.video import SparseFrameSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sparse creative video experiment.")
    parser.add_argument("--config", default="configs/smoke.json", help="Path to JSON config.")
    parser.add_argument("--no-ranker", action="store_true", help="Skip pairwise ranker training.")
    parser.add_argument(
        "--strict-model",
        action="store_true",
        help="Fail if the configured multimodal generation model is unavailable.",
    )
    parser.add_argument(
        "--strict-video",
        action="store_true",
        help="Fail if any sample falls back to deterministic synthetic frames.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing results.jsonl rows and process only missing video_ids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_file(args.config)
    set_seed(config.seed)

    output_dir = ensure_dir(PROJECT_ROOT / config.output_dir)
    samples = load_samples(config.data, project_root=PROJECT_ROOT)
    checkpoint_path = output_dir / "results.jsonl"
    existing_results = _load_existing_results(checkpoint_path) if args.resume else {}
    if not args.resume:
        checkpoint_path.write_text("", encoding="utf-8")
    if existing_results:
        samples = [sample for sample in samples if _sample_key(sample) not in existing_results]

    sampler = SparseFrameSampler(config.sampling)
    encoder, encoder_report = build_frame_encoder(config.models)
    text_reward_model = build_text_reward_model(config.models)
    script_generation_model = build_script_generation_model(config.models)
    if (
        config.models.active_text_reward_model
        and text_reward_model.model_report.get("runtime") == "openai_compatible_text_reward"
        and not text_reward_model.model_report.get("endpoint_url")
    ):
        raise RuntimeError(
            "Configured text-only reward model is active but endpoint_url is empty. "
            "Set TEXT_REWARD_ENDPOINT_URL or disable active_text_reward_model for an ablation."
        )
    generator = StructuredScriptGenerator(config.generation)
    scorer = SelfRewardScorer(config.reward, config.generation, text_reward_model)

    results = []
    preference_pairs = []
    flat_candidates = []
    for result in existing_results.values():
        results.append(result)
        for candidate in result.get("candidates", []):
            flat_candidates.append(candidate)
        preference_pairs.extend(
            build_preference_pairs(
                sample_id=result["video_id"],
                candidate_rows=result.get("candidates", []),
                margin=config.reward.preference_margin,
            )
        )

    for sample in samples:
        sample_key = _sample_key(sample)
        batch = sampler.sample(
            video_path=sample.video_path,
            video_id=sample.video_id,
            metadata={
                "caption": sample.caption,
                "category": sample.category,
                "selling_points": sample.selling_points,
            },
        )
        if args.strict_video and batch.source_kind == "synthetic":
            raise RuntimeError(
                f"Video frames were not loaded for `{sample.video_id}`. "
                f"Resolved video_path=`{sample.video_path}`. "
                "Check data.video_root/video_search_dirs or disable --strict-video for smoke runs."
            )
        representation = encoder.encode(batch, context_text=sample.context_text)
        candidates = script_generation_model.generate(sample, batch, config.generation)
        if args.strict_model and not candidates:
            raise RuntimeError(
                "Configured multimodal generation model did not return candidates. "
                "Check endpoint_url, provider, and allow_heavy_model_load."
            )
        if args.strict_model and len(candidates) < config.generation.num_candidates:
            raise RuntimeError(
                f"Configured multimodal generation model returned {len(candidates)} candidates "
                f"for `{sample.video_id}`, expected {config.generation.num_candidates}. "
                "Tighten the server prompt/parser or disable --strict-model for probe runs."
            )
        generation_source = "multimodal_model" if candidates else "rule_generator"
        if not candidates:
            candidates = generator.generate(sample, representation)
        candidate_rows = []
        for candidate in candidates:
            reward = scorer.score(candidate, representation)
            row = candidate.as_dict()
            row["video_id"] = sample.video_id
            row["reward"] = reward.as_dict()
            row["quality"] = evaluate_generation_quality(candidate, sample)
            row["generation_source"] = generation_source
            candidate_rows.append(row)
            flat_candidates.append(row)

        best = max(candidate_rows, key=lambda row: row["reward"]["total"])
        pairs = build_preference_pairs(
            sample_id=sample_key,
            candidate_rows=candidate_rows,
            margin=config.reward.preference_margin,
        )
        preference_pairs.extend(pairs)

        result_row = {
            "sample_key": sample_key,
            "video_id": sample.video_id,
            "dataset": config.data.name,
            "video_path": sample.video_path,
            "caption": sample.caption,
            "question": sample.question,
            "answer": sample.answer,
            "category": sample.category,
            "selling_points": sample.selling_points,
            "sampling": {
                "source_kind": batch.source_kind,
                "selected_indices": batch.selected_indices,
                "bin_ids": batch.bin_ids,
                **batch.metadata,
            },
            "representation": {
                "content_tags": representation.content_tags,
                "keyframe_source_indices": representation.keyframe_source_indices,
                "aggregation_weights": [
                    round(value, 6) for value in representation.aggregation_weights
                ],
                "diagnostics": representation.diagnostics,
            },
            "best_candidate": best,
            "candidates": candidate_rows,
            "generation_source": generation_source,
        }
        results.append(result_row)
        _append_jsonl(checkpoint_path, result_row)

    metrics = summarize_results(results, preference_pairs)
    ranker_report = (
        {"status": "skipped", "reason": "--no-ranker"}
        if args.no_ranker
        else train_pairwise_ranker(flat_candidates, preference_pairs)
    )
    metrics["ranker"] = ranker_report
    metrics["model_runtime"] = {
        "video_model": encoder_report,
        "script_generation_model": script_generation_model.model_report,
        "text_reward_model": text_reward_model.model_report,
    }
    metrics["config"] = config.as_dict()

    results = sorted(results, key=lambda row: (str(row.get("video_id", "")), str(row.get("sample_key", ""))))
    write_jsonl(output_dir / "results.jsonl", results)
    write_jsonl(output_dir / "preference_pairs.jsonl", preference_pairs)
    write_json(output_dir / "metrics.json", metrics)
    write_markdown_report(output_dir / "metrics_report.md", metrics)

    print(f"Loaded samples: {len(samples)}")
    print(f"Candidates: {len(flat_candidates)}")
    print(f"Preference pairs: {len(preference_pairs)}")
    print(f"Mean best reward: {metrics.get('mean_best_reward', 0.0)}")
    print(f"Control success best: {metrics.get('control_success_rate_best', 0.0)}")
    print(f"Outputs: {output_dir}")
    print_metrics_summary(metrics)


def _load_existing_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for row in iter_jsonl(path):
        sample_key = row.get("sample_key") or _sample_key_from_row(row)
        if sample_key:
            row["sample_key"] = sample_key
            rows[str(sample_key)] = row
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _sample_key(sample) -> str:
    payload = {
        "video_id": sample.video_id,
        "question": sample.question,
        "answer": sample.answer,
        "caption": sample.caption,
    }
    return _sample_key_payload(payload)


def _sample_key_from_row(row: dict) -> str:
    payload = {
        "video_id": row.get("video_id", ""),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "caption": row.get("caption", ""),
    }
    return _sample_key_payload(payload)


def _sample_key_payload(payload: dict) -> str:
    video_id = str(payload.get("video_id", ""))
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"{video_id}::{stable_hash_int(text, 10**12):012d}"


if __name__ == "__main__":
    main()
