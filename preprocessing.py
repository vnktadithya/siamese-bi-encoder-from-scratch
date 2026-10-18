"""Tokenization, vocabulary, and sentence -> token ids / bag-of-words vector."""

import numpy as np

UNK_TOKEN = '<unk>'
UNK_ID = 0


def tokenize(sentence: str) -> list[str]:
    return sentence.lower().split()


def build_vocab(pairs) -> dict[str, int]:
    # Built from training pairs only. Test and probe sentences never enter the
    # vocabulary, so the model can't get a head start on data it is scored on.
    vocab = {UNK_TOKEN: UNK_ID}
    for sentence_a, sentence_b, *_ in pairs:
        for token in tokenize(sentence_a) + tokenize(sentence_b):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def to_ids(sentence: str, vocab: dict[str, int]) -> list[int]:
    # Every out-of-vocabulary word maps to the same <unk> id.
    return [vocab.get(token, UNK_ID) for token in tokenize(sentence)]


def bow_vector(sentence: str, vocab: dict[str, int]) -> np.ndarray:
    # Out-of-vocabulary words are dropped, not counted under <unk>,
    # so slot 0 of a bag-of-words vector is always 0.
    vector = np.zeros(len(vocab), dtype=np.int64)
    for token in tokenize(sentence):
        if token in vocab:
            vector[vocab[token]] += 1
    return vector
