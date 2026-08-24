"""Shared results.json read/merge for the quantization scripts.

Each script (perplexity, throughput, MMLU) only knows about its own metric.
This merges by checkpoint label so results/results.json ends up as one row
per checkpoint with every metric filled in as each script runs, in whatever
order they're run.
"""

import json
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "results" / "results.json"


def update(label: str, metric: str, value: dict) -> None:
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    data = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}
    data.setdefault(label, {})[metric] = value
    RESULTS_FILE.write_text(json.dumps(data, indent=2))
