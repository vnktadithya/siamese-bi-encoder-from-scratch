"""Training: MSE between cosine(u, v) and the 0/1 label (SBERT's regression objective)."""

from collections.abc import Callable

import torch
from torch import optim

from config import LEARNING_RATE, LOG_EVERY, TRAIN_STEPS
from encoder import Encoder, pair_cosine


def batch_loss(encoder: Encoder, pairs, vocab: dict[str, int]) -> torch.Tensor:
    losses = [(pair_cosine(encoder, a, b, vocab) - label) ** 2 for a, b, label, _ in pairs]
    return torch.stack(losses).mean()


def train_encoder(encoder: Encoder, pairs, vocab: dict[str, int],
                  steps: int = TRAIN_STEPS, lr: float = LEARNING_RATE,
                  log: Callable[[str], None] = print) -> list[float]:
    """Full-batch Adam on every pair. Updates the encoder in place; returns the loss history."""
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    history = []
    for step in range(steps):
        # Reset gradients every step, otherwise they accumulate across steps.
        optimizer.zero_grad()
        loss = batch_loss(encoder, pairs, vocab)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if step % LOG_EVERY == 0:
            log(f'step {step:3d}  loss {loss.item():.4f}')
    return history
