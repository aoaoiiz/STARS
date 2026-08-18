from __future__ import annotations

from collections import Counter
from typing import Any

from .data import VideoSample
from .generation import ScriptCandidate
from .semantic_points import SemanticPointEvaluator, SemanticPointReference
from .utils import clip01, tokenize


def evaluate_generation_quality(
    candidate: ScriptCandidate,
    sample: VideoSample,
    semantic_point_evaluator: SemanticPointEvaluator,
    semantic_point_reference: SemanticPointReference,
) -> dict[str, Any]:
    prediction_tokens = tokenize(candidate.text)
    reference_tokens = tokenize(sample.reference_text)
    quality = {
        "cider_lite": round(
            _cider_lite(prediction_tokens, reference_tokens),
            6,
        ),
        "repetition_rate": round(_repetition_rate(prediction_tokens), 6),
    }
    quality.update(
        semantic_point_evaluator.evaluate(candidate, semantic_point_reference)
    )
    return quality


def _cider_lite(
    prediction_tokens: list[str],
    reference_tokens: list[str],
) -> float:
    if not prediction_tokens or not reference_tokens:
        return 0.0
    scores: list[float] = []
    for order in range(1, 5):
        prediction_counts = _ngram_counts(prediction_tokens, order)
        reference_counts = _ngram_counts(reference_tokens, order)
        if not prediction_counts or not reference_counts:
            scores.append(0.0)
            continue
        overlap = sum(
            min(count, reference_counts.get(ngram, 0))
            for ngram, count in prediction_counts.items()
        )
        precision = overlap / max(1, sum(prediction_counts.values()))
        recall = overlap / max(1, sum(reference_counts.values()))
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return clip01(sum(scores) / len(scores))


def _repetition_rate(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return clip01(repeated / len(tokens))


def _ngram_counts(
    tokens: list[str],
    order: int,
) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[index : index + order])
        for index in range(len(tokens) - order + 1)
    )
