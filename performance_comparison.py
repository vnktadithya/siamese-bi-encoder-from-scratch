"""Bag-of-words baseline vs untrained encoder vs trained encoder, over every seed in SEEDS.

For each seed: score train and test pairs, report per-pair cosines, AUC, and the
trained encoder's final MSE on both splits. Output is printed and saved to
logs/performance_comparison.log (UTF-8).
"""

import os
import torch

from config import LOG_DIR, SEEDS
from encoder import Encoder
from metrics import auc
from pairs import TEST_PAIRS, TRAIN_PAIRS
from preprocessing import build_vocab
from report import Report
from scoring import baseline_scores, encoder_scores, get_labels
from train import batch_loss, train_encoder

LOG_PATH = os.path.join(LOG_DIR, 'performance_comparison.log')
SPLITS = {'Train': TRAIN_PAIRS, 'Test': TEST_PAIRS}
MODELS = ['baseline', 'untrained', 'trained']


def log_split(log: Report, split: str, pairs, scores: dict[str, list[float]]) -> None:
    header = f"{'split':<7}{'category':<10}{'label':>6}" + ''.join(f'{m:>12}' for m in MODELS)
    log(header)
    log('-' * len(header))
    for i, (_, _, label, category) in enumerate(pairs):
        row = ''.join(f'{scores[m][i]:>12.4f}' for m in MODELS)
        log(f'{split:<7}{category:<10}{label:>6.1f}{row}')

    labels = get_labels(pairs)
    log(f'{split} AUC  ' + '  '.join(f'{m} {auc(scores[m], labels):.4f}' for m in MODELS))
    log()


def run_seed(seed: int, vocab: dict[str, int], baseline: dict[str, list[float]], log: Report) -> None:
    log('=' * 60)
    log(f'SEED {seed}')
    log('=' * 60)

    torch.manual_seed(seed)
    encoder = Encoder(len(vocab))
    untrained = {split: encoder_scores(pairs, encoder, vocab) for split, pairs in SPLITS.items()}

    train_encoder(encoder, TRAIN_PAIRS, vocab, log=log)   # updates the encoder in place
    trained = {split: encoder_scores(pairs, encoder, vocab) for split, pairs in SPLITS.items()}
    log()

    for split, pairs in SPLITS.items():
        scores = {'baseline': baseline[split], 'untrained': untrained[split], 'trained': trained[split]}
        log_split(log, split, pairs, scores)

    with torch.no_grad():
        train_mse = batch_loss(encoder, TRAIN_PAIRS, vocab).item()
        test_mse = batch_loss(encoder, TEST_PAIRS, vocab).item()
    log(f'Trained encoder final MSE  train {train_mse:.6f}  test {test_mse:.6f}')
    log()


def main() -> None:
    log = Report()
    vocab = build_vocab(TRAIN_PAIRS)
    # The baseline has no parameters, so it is the same for every seed.
    baseline = {split: baseline_scores(pairs, vocab) for split, pairs in SPLITS.items()}

    for seed in SEEDS:
        run_seed(seed, vocab, baseline, log)

    log.save(LOG_PATH)


if __name__ == '__main__':
    main()
