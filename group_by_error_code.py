"""Group RCA ground-truth rows by error_code, to hand-pick sentence pairs for pairs.py.

Rows that share an error_code are candidates for same-cause pairs (category A)
or same-error-different-cause pairs (category B). The "idx N" comments in
pairs.py are the example_index values printed here.

Usage:
    python group_by_error_code.py path/to/ft20_model_outputs.jsonl [--min-size 2]
"""

import argparse
import json
from collections import defaultdict

# Rows without a real error code end up with these placeholders, which say
# nothing about the cause, so grouping on them would pair unrelated rows.
IGNORED_ERROR_CODES = {'N/A', 'None', 'misc'}


def load_rows(path: str) -> list[tuple[int, dict]]:
    rows = []
    parse_failures = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                ground_truth = json.loads(record['ground_truth'].strip())
            except json.JSONDecodeError:
                parse_failures += 1
                continue
            rows.append((record['example_index'], ground_truth))

    print(f'Parsing failure count: {parse_failures}')
    return rows


def group_by_error_code(rows: list[tuple[int, dict]]) -> dict[str, list[tuple[int, str]]]:
    groups = defaultdict(list)
    for index, ground_truth in rows:
        error_code = ground_truth['error_code']
        if error_code not in IGNORED_ERROR_CODES:
            groups[error_code].append((index, ground_truth['root_cause_analysis']))

    # Largest groups first: they hold the most candidate pairs.
    return dict(sorted(groups.items(), key=lambda item: len(item[1]), reverse=True))


def print_groups(groups: dict[str, list[tuple[int, str]]], min_size: int = 2) -> None:
    for error_code, members in groups.items():
        if len(members) >= min_size:
            print(f'{error_code}: {len(members)}')
            for member in members:
                print(member)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('path', help='JSONL file with example_index and a JSON-string ground_truth per row')
    parser.add_argument('--min-size', type=int, default=2, help='smallest group to print')
    args = parser.parse_args()

    print_groups(group_by_error_code(load_rows(args.path)), args.min_size)


if __name__ == '__main__':
    main()
