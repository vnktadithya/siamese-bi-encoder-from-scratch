"""Print to the console and keep a copy, so each script can save its own log."""

import os


class Report:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, message: str = '') -> None:
        print(message)
        self.lines.append(message)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(self.lines) + '\n')
