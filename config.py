"""Experiment constants shared by every script."""

import os

EMBEDDING_DIM = 16
TRAIN_STEPS = 300
LEARNING_RATE = 0.01
LOG_EVERY = 25

SEEDS = [0, 1, 2]
OOV_PROBE_SEED = 0

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
