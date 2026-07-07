from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .data import VideoSample
from .generation import ScriptCandidate
from .utils import clip01, tokenize


def evaluate_generation_quality(
    candidate: ScriptCandidate,
    sample: VideoSample,
) -> dict[str, Any]:
    prediction = candidate.text
    reference = sample.reference_text
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    return {
        "bleu_1": round(_bleu(pred_tokens, ref_tokens, max_order=1), 6),
        "bleu_2": round(_bleu(pred_tokens, ref_tokens, max_order=2), 6),
        "bleu_4": round(_bleu(pred_tokens, ref_tokens, max_order=4), 6),
        "rouge_l": round(_rouge_l(pred_tokens, ref_tokens), 6),
        "meteor": round(_meteor(pred_tokens, ref_tokens), 6),
        "cider_lite": round(_cider_lite(pred_tokens, ref_tokens), 6),
        "distinct_1": round(_distinct(pred_tokens, order=1), 6),
        "distinct_2": round(_distinct(pred_tokens, order=2), 6),
        "repetition_rate": round(_repetition_rate(pred_tokens), 6),
        "length_tokens": len(pred_tokens),
        "reference_token_coverage": round(_reference_coverage(pred_tokens, ref_tokens), 6),
        "answer_hit": _answer_hit(prediction, sample.answer),
        "selling_point_coverage": round(_selling_point_coverage(prediction, sample.selling_points), 6),
    }


def _bleu(pred_tokens: list[str], ref_tokens: list[str], max_order: int) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    precisions = []
    for order in range(1, max_order + 1):
        pred_counts = _ngram_counts(pred_tokens, order)
        ref_counts = _ngram_counts(ref_tokens, order)
        overlap = 0
        for ngram, count in pred_counts.items():
            overlap += min(count, ref_counts.get(ngram, 0))
        total = max(1, sum(pred_counts.values()))
        precisions.append((overlap + 1.0) / (total + 1.0))
    geo_mean = math.exp(sum(math.log(value) for value in precisions) / max_order)
    brevity = min(1.0, math.exp(1.0 - len(ref_tokens) / max(1, len(pred_tokens))))
    return clip01(geo_mean * brevity)


def _rouge_l(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    precision = lcs / max(1, len(pred_tokens))
    recall = lcs / max(1, len(ref_tokens))
    if precision + recall == 0:
        return 0.0
    return clip01(2 * precision * recall / (precision + recall))


def _meteor(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    matches = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts)
    if matches == 0:
        return 0.0
    precision = matches / len(pred_tokens)
    recall = matches / len(ref_tokens)
    score = (10 * precision * recall) / max(1e-8, recall + 9 * precision)
    return clip01(score)


def _cider_lite(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    scores = []
    for order in range(1, 5):
        pred_counts = _ngram_counts(pred_tokens, order)
        ref_counts = _ngram_counts(ref_tokens, order)
        if not pred_counts or not ref_counts:
            scores.append(0.0)
            continue
        overlap = sum(min(count, ref_counts.get(ngram, 0)) for ngram, count in pred_counts.items())
        precision = overlap / max(1, sum(pred_counts.values()))
        recall = overlap / max(1, sum(ref_counts.values()))
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return clip01(sum(scores) / len(scores))


def _distinct(tokens: list[str], order: int) -> float:
    counts = _ngram_counts(tokens, order)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return clip01(len(counts) / total)


def _repetition_rate(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return clip01(repeated / len(tokens))


def _reference_coverage(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not ref_tokens:
        return 0.0
    pred_vocab = set(pred_tokens)
    ref_vocab = set(ref_tokens)
    return clip01(len(pred_vocab & ref_vocab) / max(1, len(ref_vocab)))


def _answer_hit(prediction: str, answer: str) -> float:
    if not answer:
        return 0.0
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0
    prediction_tokens = set(tokenize(prediction))
    return float(all(token in prediction_tokens for token in answer_tokens))


def _selling_point_coverage(prediction: str, selling_points: list[str]) -> float:
    points = [point for point in selling_points if point]
    if not points:
        return 0.0
    hits = sum(1 for point in points if point in prediction)
    return clip01(hits / len(points))


def _ngram_counts(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[idx : idx + order]) for idx in range(len(tokens) - order + 1))


def _lcs_len(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]
