"""The siamese encoder: embedding lookup + mean pooling, shared by both sentences."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBEDDING_DIM
from preprocessing import to_ids


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = EMBEDDING_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # One unbatched sentence: (seq_len,) -> (seq_len, dim) -> (dim,)
        return self.embedding(ids).mean(dim=0)


def encode(encoder: Encoder, sentence: str, vocab: dict[str, int]) -> torch.Tensor:
    ids = torch.tensor(to_ids(sentence, vocab), dtype=torch.long)
    return encoder(ids)


def pair_cosine(encoder: Encoder, sentence_a: str, sentence_b: str,
                vocab: dict[str, int]) -> torch.Tensor:
    # Siamese: the same encoder (same weights) embeds both sentences, and the
    # two only interact after pooling, through cosine(u, v).
    u = encode(encoder, sentence_a, vocab)
    v = encode(encoder, sentence_b, vocab)
    return F.cosine_similarity(u, v, dim=0)
