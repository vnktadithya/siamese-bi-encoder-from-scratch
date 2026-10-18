"""Invariants checked before trusting any training result.

1. Mean pooling is exactly (bag-of-words / sentence length) @ embedding table,
   for a sentence with no OOV words.
2. The pooled vector and the batch loss stay connected to the autograd graph.
3. Reference losses: the untrained encoder and the bag-of-words baseline,
   both scored with the training MSE.
"""

import torch

from config import EMBEDDING_DIM, SEEDS
from encoder import Encoder, encode
from pairs import TRAIN_PAIRS
from preprocessing import bow_vector, build_vocab, tokenize
from scoring import baseline_scores, get_labels
from train import batch_loss


def check_mean_pooling(encoder: Encoder, vocab: dict[str, int]) -> None:
    sentence = TRAIN_PAIRS[0][0]
    pooled = encode(encoder, sentence, vocab)

    bow = torch.tensor(bow_vector(sentence, vocab), dtype=torch.float32)
    expected = (bow / len(tokenize(sentence))) @ encoder.embedding.weight

    assert pooled.shape == (EMBEDDING_DIM,), pooled.shape
    assert torch.allclose(pooled, expected), 'mean pooling != BoW-weighted embedding average'
    assert pooled.requires_grad and pooled.grad_fn is not None, 'pooled vector detached from graph'
    print(f'mean pooling == (BoW / n) @ E: True | shape {tuple(pooled.shape)} | grad_fn {pooled.grad_fn}')


def check_loss_graph(encoder: Encoder, vocab: dict[str, int]) -> None:
    loss = batch_loss(encoder, TRAIN_PAIRS, vocab)
    assert loss.requires_grad and loss.grad_fn is not None, 'loss detached from graph'
    print(f'untrained encoder MSE on train pairs: {loss.item():.4f} | grad_fn {loss.grad_fn}')


def baseline_mse() -> None:
    vocab = build_vocab(TRAIN_PAIRS)
    scores = baseline_scores(TRAIN_PAIRS, vocab)
    labels = get_labels(TRAIN_PAIRS)
    mse = sum((s - y) ** 2 for s, y in zip(scores, labels, strict=True)) / len(labels)
    print(f'bag-of-words baseline MSE on train pairs: {mse:.4f}')


def main() -> None:
    vocab = build_vocab(TRAIN_PAIRS)
    torch.manual_seed(SEEDS[0])
    encoder = Encoder(len(vocab))

    check_mean_pooling(encoder, vocab)
    check_loss_graph(encoder, vocab)
    baseline_mse()


if __name__ == '__main__':
    main()
