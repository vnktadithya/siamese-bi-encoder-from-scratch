"""Toy dataset: (sentence_a, sentence_b, label, category).

label 1.0 = same root cause, 0.0 = different root cause.
Sentences are lowercase with no punctuation, because the tokenizer is a whitespace split.
"idx N" = example_index of the ground_truth root cause in the RCA fine-tuning
outputs (ft20_model_outputs.jsonl) that the sentence was written from.

The four categories cross lexical overlap with the label, so word overlap alone
can't solve the task:
    A  high overlap, same cause       (1.0)
    B  high overlap, different cause  (0.0)  <- bag-of-words scores these too high
    C  no overlap,   same cause       (1.0)  <- bag-of-words scores these 0
    D  low overlap,  different cause  (0.0)
"""

TRAIN_PAIRS = [
    # --- A: high overlap, same cause (1.0) ---
    # idx 2
    ("valueerror because split on equals had no maxsplit so values with equals broke unpacking",
     "valueerror because the field was split on every equals sign instead of only the first",
     1.0, 'A'),
    # idx 9
    ("attributeerror because redis returned str keys and the code still called decode",
     "redis returned keys as str not bytes so calling decode raised attributeerror",
     1.0, 'A'),
    # idx 18
    ("dtype mismatch because tensors were created as float32 while the model ran in fp16",
     "tensors hardcoded to float32 caused a dtype mismatch when the model ran in half precision",
     1.0, 'A'),

    # --- B: high overlap, different cause (0.0) ---
    # idx 9 vs idx 32
    ("attributeerror because redis returned str keys and the code still called decode",
     "attributeerror because the code called get partial shape on the wrapper tensor not the node",
     0.0, 'B'),
    # idx 3 vs idx 34
    ("attributeerror because a wrong object was passed as the flow run argument",
     "attributeerror because a non callback object was passed to model fit",
     0.0, 'B'),
    # idx 29 vs idx 50
    ("typeerror in hash because json dumps cannot serialize bytes attribute values",
     "typeerror in hash because the key class does not define a hash method",
     0.0, 'B'),

    # --- C: no overlap, same cause (1.0) -- written by hand, not from the RCA file ---
    ("pod restarts repeatedly with exit code 137",
     "kernel oom killer terminated process exceeding memory limit",
     1.0, 'C'),
    ("connection refused on port 5432",
     "postgres server is not running",
     1.0, 'C'),
    ("ssl certificate verify failed for internal api",
     "server presents a self signed cert missing from trust store",
     1.0, 'C'),

    # --- D: low overlap, different cause (0.0) ---
    # idx 4 vs idx 30
    ("uppercase work pool name violated lowercase block name validation",
     "cold dlopen could not resolve versioned rocm shared library",
     0.0, 'D'),
    # idx 35 vs idx 40
    ("gloo coordination across thousands of ranks caused socket timeouts",
     "optional converter called a modern converter with one argument",
     0.0, 'D'),
    # idx 10 vs idx 43
    ("swallowed async exceptions left node attributes empty downstream",
     "django serializer ran both ipv4 and ipv6 validators on input",
     0.0, 'D'),
]

# Held-out: new phrasings of causes already seen in training, built from training vocab.
# Limitation: with 12 training pairs this tests paraphrase robustness within
# known causes, NOT generalization to unseen causes.
TEST_PAIRS = [
    # A
    # idx 9
    ("the code called decode on str keys returned from redis",
     "redis keys were str so decode raised attributeerror",
     1.0, 'A'),
    # B
    # idx 34 vs idx 3
    ("attributeerror because the wrong object was passed to model fit",
     "attributeerror because the wrong object was passed as the flow run",
     0.0, 'B'),
    # C
    ("pod restarts with exit code 137",
     "process exceeding memory limit",
     1.0, 'C'),
    # D
    # idx 9 vs idx 4
    ("redis keys returned as str",
     "uppercase work pool name violated validation",
     0.0, 'D'),
]
