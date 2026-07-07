from __future__ import annotations

from typing import Any

import numpy as np


FEATURE_KEYS = ["alignment", "readability", "rhythm", "control", "risk"]


def train_pairwise_ranker(
    candidate_rows: list[dict[str, Any]],
    preference_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Train a tiny pairwise ranker from self-reward preferences.

    This is intentionally a baseline. For the paper, it gives you a reproducible
    "self-reward -> preference pairs -> ranker" path without user logs.
    """

    if not preference_pairs:
        return {"status": "skipped", "reason": "no preference pairs"}

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return {"status": "skipped", "reason": "scikit-learn is not installed"}

    by_id = {row["candidate_id"]: row for row in candidate_rows}
    features = []
    labels = []
    for pair in preference_pairs:
        chosen = by_id.get(pair["chosen_id"])
        rejected = by_id.get(pair["rejected_id"])
        if not chosen or not rejected:
            continue
        chosen_vec = _feature_vector(chosen)
        rejected_vec = _feature_vector(rejected)
        features.append(chosen_vec - rejected_vec)
        labels.append(1)
        features.append(rejected_vec - chosen_vec)
        labels.append(0)

    if len(set(labels)) < 2 or not features:
        return {"status": "skipped", "reason": "not enough pairwise labels"}

    x = np.stack(features, axis=0)
    y = np.asarray(labels)
    model = LogisticRegression(random_state=0, max_iter=200).fit(x, y)
    accuracy = float(model.score(x, y))
    return {
        "status": "trained",
        "num_pairs": len(preference_pairs),
        "num_training_rows": int(len(y)),
        "train_accuracy": round(accuracy, 6),
        "feature_keys": FEATURE_KEYS,
        "coef": [round(float(value), 6) for value in model.coef_[0]],
        "intercept": round(float(model.intercept_[0]), 6),
    }


def _feature_vector(row: dict[str, Any]) -> np.ndarray:
    reward = row["reward"]
    return np.asarray([reward[key] for key in FEATURE_KEYS], dtype=np.float32)
