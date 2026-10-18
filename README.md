# Siamese Bi-Encoder from Scratch

A toy version of the Sentence-BERT siamese setup (Reimers & Gurevych, 2019), built to see what the training objective does to an embedding space with nothing pretrained underneath it. One encoder, a trainable embedding table with mean pooling, embeds both sentences of a pair with shared weights. The two sentences only interact after pooling, through `cosine(u, v)`. Training uses SBERT's regression objective: MSE between `cosine(u, v)` and a 0/1 label for "same root cause".

The sentences are root-cause descriptions of real bugs, taken from the RCA dataset used in my QLoRA fine-tuning study.

## How it works

1. **Vocabulary** from training pairs only, with whitespace tokenization. Every out-of-vocabulary word maps to a single `<unk>` id (0).
2. **Encoder:** `nn.Embedding(vocab_size, 16)` followed by a mean over tokens. One unbatched sentence goes in, one 16-d vector comes out.
3. **Loss:** `mean((cosine(u, v) - label)^2)` over all 12 training pairs, full-batch Adam, lr 0.01, 300 steps.
4. **Baseline:** cosine similarity of bag-of-words count vectors, which measures word overlap and nothing else.
5. **Evaluation:** per-pair cosines and AUC (the probability that a random same-cause pair outscores a random different-cause pair, with ties counted as 0.5), for the baseline, the untrained encoder, and the trained encoder, across seeds 0, 1 and 2.

## Dataset design

The pairs cross lexical overlap with the label, so word overlap alone can't solve the task:

| Category | Overlap | Label | Why it's there |
|---|---|---|---|
| A | high | same cause (1.0) | the easy positive |
| B | high | different cause (0.0) | bag-of-words scores these too high |
| C | none | same cause (1.0) | bag-of-words scores these exactly 0 |
| D | low | different cause (0.0) | the easy negative |

There are 12 training pairs (3 per category) and 4 test pairs (1 per category). Test pairs rephrase causes that appear in training, using only training vocabulary. They test paraphrase robustness within known causes, **not** generalization to unseen causes. Category C pairs were written by hand. The rest come from `example_index` rows of the RCA dataset (the `idx N` comments in `pairs.py`), which were selected with `group_by_error_code.py`.

## Results

From `logs/performance_comparison.log`:

| Seed | Train AUC (baseline / untrained / trained) | Test AUC (baseline / untrained / trained) | Trained MSE (train / test) |
|---|---|---|---|
| 0 | 0.5139 / 0.3611 / 1.0 | 0.375 / 0.50 / 1.0 | 0.000129 / 0.1056 |
| 1 | 0.5139 / 0.5278 / 1.0 | 0.375 / 0.25 / 1.0 | 0.000043 / 0.0649 |
| 2 | 0.5139 / 0.6944 / 1.0 | 0.375 / 0.75 / 1.0 | 0.000085 / 0.0635 |

- **The baseline fails on exactly what it was built to fail on.** Its test AUC of 0.375 comes from the B pair (high overlap, different cause) scoring 0.70, above both positives, and the C pair (no overlap, same cause) scoring 0.0.
- **The untrained encoder is noise.** Its AUC ranges from 0.36 to 0.69 on train and 0.25 to 0.75 on test depending on the seed. That spread is the noise floor any trained result has to clear.
- **Perfect AUC hides a generalization gap.** The trained encoder reaches 1.0 AUC on both splits for every seed. Its test MSE, however, is 750 to 1,500 times its train MSE. AUC only checks ranking, and with 2 positives × 2 negatives the test AUC rests on just 4 comparisons. The positives still rank above the negatives, but the test cosines sit far from their targets (seed 0: 0.74 and 0.71 for positives, 0.40 and −0.33 for negatives).

## OOV probe

`oov_probe.py` retrains the seed-0 encoder and scores pairs in which words are progressively swapped for unseen synonyms. From `logs/oov_probe.log`:

| Group | Label | OOV (a / b) | Baseline | Trained |
|---|---|---|---|---|
| POS-0 | 1.0 | 0/10, 0/8 | 0.4472 | 0.7395 |
| POS-1 | 1.0 | 1/10, 1/8 | 0.5040 | 0.8409 |
| POS-2 | 1.0 | 4/10, 2/8 | 0.5000 | 0.7928 |
| POS-3 | 1.0 | 7/11, 7/9 | 0.0000 | 0.9710 |
| NEG-0 | 0.0 | 0/5, 0/6 | 0.0000 | −0.3314 |
| NEG-1 | 0.0 | 1/5, 2/7 | 0.0000 | −0.0344 |
| NEG-2 | 0.0 | 3/5, 5/6 | 0.0000 | 0.8561 |
| FULL-NEG | 0.0 | 6/6, 7/7 | 0.0000 | 1.0000 |
| FULL-POS | 1.0 | 5/5, 5/5 | 0.0000 | 1.0000 |

- **The `<unk>` row is never trained.** No training sentence contains an OOV word, so row 0 gets no gradient. The probe confirms it is bit-identical before and after training.
- **OOV words push every pair toward cosine 1, whatever the label.** All OOV words share one `<unk>` vector, so the more of a sentence is OOV, the more its mean-pooled embedding becomes that one vector. When both sentences are fully OOV, they pool to the identical vector, and FULL-NEG and FULL-POS both score exactly 1.0. NEG-2 climbing from −0.33 to 0.86 is the same effect. So POS-3's 0.97 is not a success. It is what any heavily-OOV pair would score.
- **The baseline drops to 0 instead,** because the bag-of-words vector discards OOV words.

This is the gap pretraining fills. Real SBERT starts from BERT's subword vocabulary and pretrained representations, so an unseen word still decomposes into pieces that carry meaning. This toy model has neither, so anything outside its 12 training pairs is untouched random initialization.

## Project structure

```
├── config.py                  # embedding dim, steps, lr, seeds, log dir
├── pairs.py                   # TRAIN_PAIRS / TEST_PAIRS: (sentence_a, sentence_b, label, category)
├── oov_pairs.py               # OOV_PAIRS for the probe, kept separate from the main evaluation
├── preprocessing.py           # tokenize, build_vocab, to_ids, bow_vector
├── encoder.py                 # Encoder (embedding + mean pool), encode, pair_cosine
├── metrics.py                 # NumPy cosine similarity, AUC
├── scoring.py                 # baseline_scores, encoder_scores, get_labels
├── train.py                   # batch_loss (MSE on cosine), train_encoder
├── report.py                  # print + save a script's output as a UTF-8 log
├── performance_comparison.py  # baseline vs untrained vs trained, all seeds -> logs/
├── oov_probe.py               # OOV probe on the seed-0 encoder -> logs/
├── sanity_checks.py           # mean-pooling equivalence and autograd-graph checks
├── group_by_error_code.py     # groups RCA rows by error_code, used to pick source pairs
├── logs/
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

python sanity_checks.py            # run first
python performance_comparison.py   # writes logs/performance_comparison.log
python oov_probe.py                # writes logs/oov_probe.log
```

`group_by_error_code.py` needs the RCA model-output JSONL, which is not included in this repo:

```bash
python group_by_error_code.py path/to/ft20_model_outputs.jsonl --min-size 2
```

## Limitations

- 12 training pairs and 4 test pairs. Test AUC moves in steps of 0.25, and every test pair reuses a cause seen in training.
- There is no pretraining, so OOV words collapse to one random vector (see the OOV probe).
- Mean pooling is order-blind and can't represent negation. "x is not y" and "x is y" differ only by the averaged-in vector for `not`.
- Whitespace tokenization with no punctuation handling. An empty sentence would make the mean pool NaN, and it isn't guarded against because no pair contains one.
- The objective is SBERT's 2019 regression loss. Current embedding models are trained contrastively (e.g. Multiple Negatives Ranking Loss) on far more data.
