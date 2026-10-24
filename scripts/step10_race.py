"""Run Treatment 7, Step 10: race against the baseline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.datasets import get_dataset, read_rows
from src.core.race import race


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["x264", "nasa93dem", "health_closedissues"])
parser.add_argument("--budget", type=int, default=20)
args = parser.parse_args()

spec = get_dataset(args.dataset)
print(json.dumps(race(spec, read_rows(spec), budget=args.budget), indent=2))
