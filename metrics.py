"""Cosine similarity (NumPy, for the bag-of-words baseline) and pairwise AUC."""

import numpy as np


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    # A zero vector has no direction; score it as 0 instead of dividing by 0.
    if not np.any(u) or not np.any(v):
        return 0.0
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def auc(scores: list[float], labels: list[float]) -> float:
    """Probability that a random positive pair outscores a random negative pair.

    Every (positive, negative) combination is compared; ties count as half a win.
    """
    positives = [score for score, label in zip(scores, labels, strict=True) if label == 1.0]
    negatives = [score for score, label in zip(scores, labels, strict=True) if label != 1.0]
    if not positives or not negatives:
        raise ValueError('AUC needs at least one positive and one negative pair')

    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))
