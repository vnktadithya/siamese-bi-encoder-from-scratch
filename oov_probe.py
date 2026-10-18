"""OOV probe for the trained encoder.

Retrains the encoder exactly as performance_comparison.py does (same vocab,
seed, steps, lr), checks that the <unk> embedding row was never updated, then
scores OOV_PAIRS with the bag-of-words baseline and the trained encoder.
Output is printed and saved to logs/oov_probe.log (UTF-8).
"""

import os

import torch

from config import LOG_DIR, OOV_PROBE_SEED
from encoder import Encoder
from oov_pairs import OOV_PAIRS
from pairs import TRAIN_PAIRS
from preprocessing import UNK_ID, build_vocab, tokenize
from report import Report
from scoring import baseline_scores, encoder_scores
from train import train_encoder

LOG_PATH = os.path.join(LOG_DIR, 'oov_probe.log')


def oov_ratio(sentence: str, vocab: dict[str, int]) -> str:
    tokens = tokenize(sentence)
    unseen = sum(1 for token in tokens if token not in vocab)
    return f'{unseen}/{len(tokens)}'


def main() -> None:
    log = Report()
    vocab = build_vocab(TRAIN_PAIRS)

    torch.manual_seed(OOV_PROBE_SEED)
    encoder = Encoder(len(vocab))
    unk_before = encoder.embedding.weight[UNK_ID].detach().clone()

    train_encoder(encoder, TRAIN_PAIRS, vocab, log=lambda _: None)   # training curve is in the main log

    # No training sentence contains an OOV word, so the <unk> row should get
    # no gradient and stay exactly at its random initial value.
    unk_after = encoder.embedding.weight[UNK_ID].detach()
    log(f'Seed {OOV_PROBE_SEED} | vocab size {len(vocab)}')
    log(f'<unk> row unchanged by training: {torch.equal(unk_before, unk_after)}')
    log()

    baseline = baseline_scores(OOV_PAIRS, vocab)
    trained = encoder_scores(OOV_PAIRS, encoder, vocab)

    header = f"{'group':<10}{'label':>6}{'oov_a':>8}{'oov_b':>8}{'baseline':>11}{'trained':>11}"
    log(header)
    log('-' * len(header))
    for (a, b, label, group), base_score, trained_score in zip(OOV_PAIRS, baseline, trained, strict=True):
        log(f'{group:<10}{label:>6.1f}{oov_ratio(a, vocab):>8}{oov_ratio(b, vocab):>8}'
            f'{base_score:>11.4f}{trained_score:>11.4f}')

    log()
    log('Reproducibility: POS-0 and NEG-0 are TEST_PAIRS A and D verbatim, so their trained')
    log(f'cosines must equal the seed-{OOV_PROBE_SEED} Test rows in performance_comparison.log.')

    log.save(LOG_PATH)


if __name__ == '__main__':
    main()
