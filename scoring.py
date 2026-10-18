"""Score every pair with the bag-of-words baseline or an encoder."""

import torch

from encoder import Encoder, pair_cosine
from metrics import cosine_similarity
from preprocessing import bow_vector


def get_labels(pairs) -> list[float]:
    return [label for _, _, label, _ in pairs]


def baseline_scores(pairs, vocab: dict[str, int]) -> list[float]:
    return [cosine_similarity(bow_vector(a, vocab), bow_vector(b, vocab))
            for a, b, _, _ in pairs]


def encoder_scores(pairs, encoder: Encoder, vocab: dict[str, int]) -> list[float]:
    with torch.no_grad():
        return [pair_cosine(encoder, a, b, vocab).item() for a, b, _, _ in pairs]
