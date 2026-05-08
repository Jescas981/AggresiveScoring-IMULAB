import sys
import time


class ProgressBar:
    def __init__(self, total, label=""):
        self.total = total
        self.label = label
        self.start_time = time.time()

    def update(self, i):
        pct = 100 * i / self.total if self.total else 100
        elapsed = time.time() - self.start_time

        bar_len = 30
        filled = int(bar_len * pct / 100)

        bar = "█" * filled + "░" * (bar_len - filled)

        sys.stdout.write(
            f"\r{self.label} |{bar}| {pct:5.1f}% ({i}/{self.total})"
        )
        sys.stdout.flush()

    def close(self):
        print()