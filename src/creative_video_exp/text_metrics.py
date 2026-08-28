from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any

from .data import VideoSample
from .generation import ScriptCandidate
from .semantic_points import SemanticPointEvaluator, SemanticPointReference
from .utils import clip01, tokenize


def evaluate_generation_quality(
    candidate: ScriptCandidate,
    sample: VideoSample,
    semantic_point_evaluator: SemanticPointEvaluator | None = None,
    semantic_point_reference: SemanticPointReference | None = None,
) -> dict[str, Any]:
    prediction = candidate.text
    reference = sample.reference_text
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    cider_reference_texts = _cider_lite_reference_texts(sample)
    cider_reference_scores = [
        _cider_lite(pred_tokens, tokenize(reference_text))
        for reference_text in cider_reference_texts
    ]
    cider_lite = (
        sum(cider_reference_scores) / len(cider_reference_scores)
        if cider_reference_scores
        else 0.0
    )
    language_tokens = tokenize(
        " ".join(
            [
                candidate.text,
                *(segment.salient_point for segment in candidate.timeline),
            ]
        )
    )
    overlay_audit = _on_screen_text_unicode_metrics(candidate)
    quality = {
        "bleu_1": round(_bleu(pred_tokens, ref_tokens, max_order=1), 6),
        "bleu_2": round(_bleu(pred_tokens, ref_tokens, max_order=2), 6),
        "bleu_4": round(_bleu(pred_tokens, ref_tokens, max_order=4), 6),
        "rouge_l": round(_rouge_l(pred_tokens, ref_tokens), 6),
        "meteor": round(_meteor(pred_tokens, ref_tokens), 6),
        "cider_lite": round(cider_lite, 6),
        "cider_lite_reference_count": len(cider_reference_texts),
        "cider_lite_reference_aggregation": (
            "equal_mean_over_evaluation_references"
            if len(sample.evaluation_references) > 1
            else "single_legacy_reference"
        ),
        "cider_lite_reference_scores": [
            round(score, 6) for score in cider_reference_scores
        ],
        "distinct_1": round(_distinct(pred_tokens, order=1), 6),
        "distinct_2": round(_distinct(pred_tokens, order=2), 6),
        "repetition_rate": round(_repetition_rate(pred_tokens), 6),
        "length_tokens": len(pred_tokens),
        "reference_token_coverage": round(_reference_coverage(pred_tokens, ref_tokens), 6),
        "answer_hit": _answer_hit(prediction, sample.answer),
        "english_token_rate": round(_english_token_rate(language_tokens), 6),
        **overlay_audit,
    }
    if semantic_point_evaluator is not None and semantic_point_reference is not None:
        quality.update(
            semantic_point_evaluator.evaluate(candidate, semantic_point_reference)
        )
    else:
        quality.update(
            {
                "semantic_point_coverage": None,
                "semantic_point_reference_available": 0.0,
                "semantic_point_reference_count": 0,
                "semantic_point_covered_count": 0,
                "semantic_point_similarity_threshold": None,
                "semantic_point_best_similarities": [],
                "semantic_point_best_segment_indices": [],
            }
        )
    return quality


def _cider_lite_reference_texts(sample: VideoSample) -> list[str]:
    if len(sample.evaluation_references) <= 1:
        return [sample.reference_text]
    return [
        _evaluation_reference_text(reference)
        for reference in sample.evaluation_references
    ]


def _evaluation_reference_text(reference: dict[str, Any]) -> str:
    semantic_points = reference.get("semantic_points", [])
    if not isinstance(semantic_points, list):
        semantic_points = []
    parts = [
        reference.get("caption", ""),
        reference.get("question", ""),
        reference.get("answer", ""),
        " ".join(str(point) for point in semantic_points if point),
        reference.get("category", ""),
    ]
    return " ".join(str(part) for part in parts if part)


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


def _english_token_rate(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    english = sum(1 for token in tokens if re.search(r"[a-z]", token, flags=re.IGNORECASE))
    return clip01(english / len(tokens))


def _on_screen_text_unicode_metrics(candidate: ScriptCandidate) -> dict[str, float | int]:
    total_letters = 0
    non_latin_letters = 0
    for segment in candidate.timeline:
        for character in segment.on_screen_text:
            if not character.isalpha():
                continue
            total_letters += 1
            if "LATIN" not in unicodedata.name(character, ""):
                non_latin_letters += 1
    return {
        "on_screen_text_has_non_latin": float(non_latin_letters > 0),
        "on_screen_text_non_latin_letter_count": non_latin_letters,
        "on_screen_text_non_latin_letter_rate": round(
            non_latin_letters / max(1, total_letters), 6
        ),
    }


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
